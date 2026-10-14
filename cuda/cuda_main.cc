#include "q38/cuda_backend.h"
#include "q38/rpc.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_tokens(const std::vector<std::int32_t>& tokens) {
    std::cout << '[';
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << tokens[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    std::string tp_root;
    std::string stage0;
    std::string stage1;
    std::string ple;
    std::string socket_path;
    std::string identity_path;
    std::string snapshot_journal_path;
    q38::ExecutorOptions options;
    std::uint64_t ple_cache_gib = 8;
    std::uint64_t ple_ram_budget_gib = 64;
    q38::PleIoModeV1 ple_io_mode = q38::PleIoModeV1::kIoUringDirect;
    std::uint32_t ple_queue_depth = 64;
    bool enable_mtp = false;
    bool enable_p2p = false;
    bool stream_ordered_p2p = false;
    bool enable_piecewise_decode_graph = false;
    bool enable_logit_diagnostics = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--tp2-root" && index + 1 < argc)
            tp_root = argv[++index];
        else if (argument == "--stage0-index" && index + 1 < argc)
            stage0 = argv[++index];
        else if (argument == "--stage1-index" && index + 1 < argc)
            stage1 = argv[++index];
        else if (argument == "--ple-layout" && index + 1 < argc)
            ple = argv[++index];
        else if (argument == "--chunk" && index + 1 < argc)
            options.append_chunk_tokens =
                static_cast<std::uint32_t>(std::stoul(argv[++index]));
        else if (argument == "--ple-cache-gib" && index + 1 < argc)
            ple_cache_gib = std::stoull(argv[++index]);
        else if (argument == "--ple-ram-budget-gib" && index + 1 < argc)
            ple_ram_budget_gib = std::stoull(argv[++index]);
        else if (argument == "--ple-io" && index + 1 < argc) {
            const std::string mode = argv[++index];
            if (mode == "direct")
                ple_io_mode = q38::PleIoModeV1::kIoUringDirect;
            else if (mode == "auto")
                ple_io_mode = q38::PleIoModeV1::kAuto;
            else if (mode == "buffered")
                ple_io_mode = q38::PleIoModeV1::kBuffered;
            else if (mode == "ram")
                ple_io_mode = q38::PleIoModeV1::kRam;
            else
                throw std::invalid_argument("unknown PLE I/O mode");
        } else if (argument == "--ple-queue-depth" && index + 1 < argc)
            ple_queue_depth =
                static_cast<std::uint32_t>(std::stoul(argv[++index]));
        else if (argument == "--socket" && index + 1 < argc)
            socket_path = argv[++index];
        else if (argument == "--session-hash" && index + 1 < argc)
            options.session_hash = std::stoull(argv[++index], nullptr, 0);
        else if (argument == "--identity" && index + 1 < argc)
            identity_path = argv[++index];
        else if (argument == "--snapshot-journal" && index + 1 < argc)
            snapshot_journal_path = argv[++index];
        else if (argument == "--context-limit" && index + 1 < argc)
            options.context_limit =
                static_cast<std::uint32_t>(std::stoul(argv[++index]));
        else if (argument == "--sampling" && index + 1 < argc) {
            const std::string mode = argv[++index];
            if (mode == "greedy")
                options.sampling.mode = q38::SamplingModeV1::kGreedy;
            else if (mode == "top-k-top-p")
                options.sampling.mode = q38::SamplingModeV1::kTopKTopP;
            else
                throw std::invalid_argument("unknown sampling mode");
        } else if (argument == "--temperature" && index + 1 < argc)
            options.sampling.temperature = std::stof(argv[++index]);
        else if (argument == "--top-p" && index + 1 < argc)
            options.sampling.top_p = std::stof(argv[++index]);
        else if (argument == "--top-k" && index + 1 < argc)
            options.sampling.top_k =
                static_cast<std::uint32_t>(std::stoul(argv[++index]));
        else if (argument == "--sampling-seed" && index + 1 < argc)
            options.sampling.seed = std::stoull(argv[++index], nullptr, 0);
        else if (argument == "--repetition-penalty" && index + 1 < argc)
            options.sampling.repetition_penalty = std::stof(argv[++index]);
        else if (argument == "--frequency-penalty" && index + 1 < argc)
            options.sampling.frequency_penalty = std::stof(argv[++index]);
        else if (argument == "--presence-penalty" && index + 1 < argc)
            options.sampling.presence_penalty = std::stof(argv[++index]);
        else if (argument == "--stop-token-id" && index + 1 < argc)
            options.stop_token_ids.push_back(
                static_cast<std::int32_t>(std::stol(argv[++index])));
        else if (argument == "--tp-p2p")
            enable_p2p = true;
        else if (argument == "--tp-p2p-async")
            stream_ordered_p2p = true;
        else if (argument == "--tp-batched-mtp") {
            enable_mtp = true;
            options.enable_batched_mtp = true;
        }
        else if (argument == "--enable-mtp")
            enable_mtp = true;
        else if (argument == "--enable-piecewise-decode-graph")
            enable_piecewise_decode_graph = true;
        else if (argument == "--enable-adaptive-mtp")
            options.enable_adaptive_mtp = true;
        else if (argument == "--enable-compact-sampling")
            options.enable_compact_sampling = true;
        else if (argument == "--enable-logit-diagnostics") {
            enable_logit_diagnostics = true;
            options.retain_last_logits_for_diagnostics = true;
        } else {
            std::cerr << "usage: q38-cuda-runtime --stage0-index FILE "
                         "--stage1-index FILE --ple-layout FILE "
                         "[--chunk N] [--ple-cache-gib N] "
                         "[--ple-io direct|auto|buffered|ram] [--ple-ram-budget-gib N] "
                         "[--ple-queue-depth N] "
                         "[--context-limit N] [--session-hash N] "
                         "[--socket PATH] --identity FILE "
                         "[--sampling greedy|top-k-top-p] "
                         "[--snapshot-journal FILE] [--stop-token-id N] "
                         "[--enable-mtp] [--tp-batched-mtp] "
                         "[--enable-piecewise-decode-graph] "
                         "[--enable-logit-diagnostics] [--enable-adaptive-mtp] [--enable-compact-sampling] "
                         "[--tp2-root DIR (experimental, replaces stage indexes)] "
                         "[--tp-p2p] [--tp-p2p-async] "
                         "[sampling options]\n";
            return 2;
        }
    }
    if ((tp_root.empty() && (stage0.empty() || stage1.empty())) || ple.empty() ||
        identity_path.empty()) {
        std::cerr << "q38-cuda-runtime: artifact and identity paths are required\n";
        return 2;
    }
    try {
        if (ple_cache_gib > (UINT64_MAX >> 30u) || ple_ram_budget_gib == 0 ||
            ple_ram_budget_gib > (UINT64_MAX >> 30u))
            throw std::invalid_argument("PLE memory budget is invalid");
        if (enable_p2p && tp_root.empty())
            throw std::invalid_argument("--tp-p2p requires --tp2-root");
        if (stream_ordered_p2p && !enable_p2p)
            throw std::invalid_argument("--tp-p2p-async requires --tp-p2p");
        options.identity = q38::load_session_identity(identity_path);
        if (options.enable_batched_mtp && tp_root.empty())
            throw std::invalid_argument("--tp-batched-mtp requires --tp2-root");
        if (!tp_root.empty() && ((enable_mtp && !options.enable_batched_mtp) || enable_piecewise_decode_graph ||
                (options.enable_adaptive_mtp && !options.enable_batched_mtp) || options.enable_compact_sampling))
            throw std::invalid_argument("TP2 requires its dedicated MTP/graph execution switches");
        auto executor = !tp_root.empty() ? q38::make_cuda_tp_executor(
            tp_root, ple, options, ple_cache_gib << 30u, ple_io_mode, ple_queue_depth,
            enable_p2p, ple_ram_budget_gib << 30u, stream_ordered_p2p)
            : q38::make_cuda_executor(
            stage0, stage1, ple, options, 0, 1, ple_cache_gib << 30u,
            ple_io_mode, ple_queue_depth, enable_mtp,
            enable_piecewise_decode_graph, 1, 0, ple_ram_budget_gib << 30u);
        const auto* tp_graph_env = std::getenv("Q38_TP_PIECEWISE_GRAPH");
        const bool configured_graph = enable_piecewise_decode_graph ||
            (!tp_root.empty() && tp_graph_env && std::string(tp_graph_env) == "1");
        std::cout << "{\"type\":\"ready\",\"backend\":\"cuda\","
                     "\"contract_version\":1,\"rpc_version\":1,"
                  << "\"tp_world_size\":" << (tp_root.empty() ? 1 : 2)
                  << ",\"tp_transport\":\"" << (enable_p2p ? "p2p" : "host") << "\""
                  << ",\"tp_stream_ordered\":" << (stream_ordered_p2p ? "true" : "false")
                  << ",\"ple_storage\":\"" << (ple_io_mode == q38::PleIoModeV1::kRam ? "ram" : "ssd") << "\""
                  << ",\"mtp_enabled\":"
                  << (enable_mtp ? "true" : "false")
                  << ",\"piecewise_decode_graph_enabled\":"
                  << (configured_graph ? "true" : "false")
                  << ",\"logit_diagnostics_enabled\":"
                  << (enable_logit_diagnostics ? "true" : "false")
                  << "}\n"
                  << std::flush;
        if (!socket_path.empty()) {
            q38::ExecutorRpcServiceV1 service(executor.get(),
                                              options.session_hash,
                                              snapshot_journal_path,
                                              enable_logit_diagnostics);
            q38::serve_executor_rpc_unix(&service, socket_path);
            return 0;
        }
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream input(line);
            std::string command;
            input >> command;
            if (command.empty()) continue;
            if (command == "quit" || command == "exit") break;
            if (command == "state") {
                const auto& frontier = executor->frontiers();
                std::cout << "{\"type\":\"state\",\"canonical\":"
                          << frontier.canonical << ",\"target\":"
                          << frontier.target << ",\"epoch\":"
                          << frontier.epoch << "}\n";
            } else if (command == "stats") {
                const auto stats = executor->stats();
                std::cout << "{\"type\":\"stats\",\"transactions\":"
                          << stats.transactions
                          << ",\"evaluated_tokens\":"
                          << stats.evaluated_tokens
                          << ",\"published_tokens\":"
                          << stats.published_tokens
                          << ",\"drafted_tokens\":" << stats.drafted_tokens
                          << ",\"rollbacks\":" << stats.rollbacks
                          << ",\"failures\":" << stats.failures
                          << ",\"cancellations\":" << stats.cancellations
                          << ",\"deadline_exceeded\":"
                          << stats.deadline_exceeded
                          << ",\"sampled_tokens\":" << stats.sampled_tokens
                          << ",\"stage0_execute_ns\":"
                          << stats.stage0_execute_ns
                          << ",\"stage1_execute_ns\":"
                          << stats.stage1_execute_ns
                          << ",\"backend_commit_ns\":"
                          << stats.backend_commit_ns
                          << ",\"draft_ns\":" << stats.draft_ns
                          << ",\"sampling_ns\":" << stats.sampling_ns
                          << "}\n";
            } else if (command == "append") {
                std::vector<std::int32_t> tokens;
                std::int64_t token = 0;
                while (input >> token)
                    tokens.push_back(static_cast<std::int32_t>(token));
                executor->append(tokens);
                std::cout << "{\"type\":\"append\",\"count\":"
                          << tokens.size() << "}\n";
            } else if (command == "seed") {
                std::cout << "{\"type\":\"tokens\",\"tokens\":["
                          << executor->seed_decode() << "]}\n";
            } else if (command == "decode") {
                std::uint32_t steps = 1;
                input >> steps;
                for (std::uint32_t step = 0; step < steps; ++step) {
                    const auto result = executor->decode_one();
                    std::cout << "{\"type\":\"tokens\",\"mode\":\"plain\","
                                 "\"tokens\":";
                    print_tokens(result.published_tokens);
                    std::cout << "}\n";
                }
            } else if (command == "spec") {
                std::uint32_t steps = 1;
                std::uint32_t width = 3;
                input >> steps >> width;
                for (std::uint32_t step = 0; step < steps; ++step) {
                    const auto result = executor->speculative_step(width);
                    std::cout << "{\"type\":\"tokens\",\"mode\":\"mtp\","
                                 "\"tokens\":";
                    print_tokens(result.published_tokens);
                    std::cout << "}\n";
                }
            } else {
                std::cout << "{\"type\":\"error\","
                             "\"message\":\"unknown command\"}\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "q38-cuda-runtime: " << error.what() << '\n';
        return 1;
    }
}
