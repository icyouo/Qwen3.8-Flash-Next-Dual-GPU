#include "q38/cuda_backend.h"
#include "q38/rpc.h"

#include <cstdint>
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
    std::string stage0;
    std::string stage1;
    std::string ple;
    std::string socket_path;
    std::string identity_path;
    std::string snapshot_journal_path;
    q38::ExecutorOptions options;
    std::uint64_t ple_cache_gib = 8;
    q38::PleIoModeV1 ple_io_mode = q38::PleIoModeV1::kIoUringDirect;
    std::uint32_t ple_queue_depth = 64;
    bool enable_mtp = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--stage0-index" && index + 1 < argc)
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
        else if (argument == "--ple-io" && index + 1 < argc) {
            const std::string mode = argv[++index];
            if (mode == "direct")
                ple_io_mode = q38::PleIoModeV1::kIoUringDirect;
            else if (mode == "auto")
                ple_io_mode = q38::PleIoModeV1::kAuto;
            else if (mode == "buffered")
                ple_io_mode = q38::PleIoModeV1::kBuffered;
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
        else if (argument == "--enable-mtp")
            enable_mtp = true;
        else {
            std::cerr << "usage: q38-cuda-runtime --stage0-index FILE "
                         "--stage1-index FILE --ple-layout FILE "
                         "[--chunk N] [--ple-cache-gib N] "
                         "[--ple-io direct|auto|buffered] "
                         "[--ple-queue-depth N] "
                         "[--context-limit N] [--session-hash N] "
                         "[--socket PATH] --identity FILE "
                         "[--sampling greedy|top-k-top-p] "
                         "[--snapshot-journal FILE] [--stop-token-id N] "
                         "[--enable-mtp] "
                         "[sampling options]\n";
            return 2;
        }
    }
    if (stage0.empty() || stage1.empty() || ple.empty() ||
        identity_path.empty()) {
        std::cerr << "q38-cuda-runtime: artifact and identity paths are required\n";
        return 2;
    }
    try {
        options.identity = q38::load_session_identity(identity_path);
        auto executor = q38::make_cuda_executor(
            stage0, stage1, ple, options, 0, 1, ple_cache_gib << 30u,
            ple_io_mode, ple_queue_depth, enable_mtp);
        std::cout << "{\"type\":\"ready\",\"backend\":\"cuda\","
                     "\"contract_version\":1,\"rpc_version\":1,"
                  << "\"mtp_enabled\":"
                  << (enable_mtp ? "true" : "false") << "}\n"
                  << std::flush;
        if (!socket_path.empty()) {
            q38::ExecutorRpcServiceV1 service(executor.get(),
                                              options.session_hash,
                                              snapshot_journal_path);
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
