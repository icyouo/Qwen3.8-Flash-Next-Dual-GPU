#include "q38/artifact.h"
#include "q38/executor.h"
#include "q38/model_plan.h"
#include "q38/ple.h"
#include "q38/ple_backend.h"
#include "q38/rpc.h"
#include "q38/mock_backend.h"
#include "q38/tensor_index.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

void print_tokens(const std::vector<std::int32_t>& tokens) {
    std::cout << '[';
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << tokens[i];
    }
    std::cout << ']';
}

void print_state(const q38::DualStageExecutor& executor) {
    const auto& f = executor.frontiers();
    std::cout << "{\"type\":\"state\",\"canonical\":" << f.canonical
              << ",\"target\":" << f.target
              << ",\"stage0\":" << f.stage0
              << ",\"stage1\":" << f.stage1
              << ",\"draft\":" << f.draft
              << ",\"epoch\":" << f.epoch
              << ",\"poisoned\":" << (executor.poisoned() ? "true" : "false")
              << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    q38::ExecutorOptions options;
    std::string manifest_path;
    std::string stage0_index_path;
    std::string stage1_index_path;
    std::string ple_layout_path;
    std::string socket_path;
    std::string identity_path;
    std::string snapshot_journal_path;
    std::uint64_t ple_cache_mib = 0;
    bool enable_logit_diagnostics = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--chunk" && i + 1 < argc) {
            options.append_chunk_tokens =
                static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--stage0-index" && i + 1 < argc) {
            stage0_index_path = argv[++i];
        } else if (arg == "--stage1-index" && i + 1 < argc) {
            stage1_index_path = argv[++i];
        } else if (arg == "--ple-layout" && i + 1 < argc) {
            ple_layout_path = argv[++i];
        } else if (arg == "--ple-cache-mib" && i + 1 < argc) {
            ple_cache_mib = std::stoull(argv[++i]);
        } else if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (arg == "--session-hash" && i + 1 < argc) {
            options.session_hash = std::stoull(argv[++i], nullptr, 0);
        } else if (arg == "--identity" && i + 1 < argc) {
            identity_path = argv[++i];
        } else if (arg == "--snapshot-journal" && i + 1 < argc) {
            snapshot_journal_path = argv[++i];
        } else if (arg == "--context-limit" && i + 1 < argc) {
            options.context_limit =
                static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--sampling" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "greedy")
                options.sampling.mode = q38::SamplingModeV1::kGreedy;
            else if (mode == "top-k-top-p")
                options.sampling.mode = q38::SamplingModeV1::kTopKTopP;
            else
                throw std::invalid_argument("unknown sampling mode");
        } else if (arg == "--temperature" && i + 1 < argc) {
            options.sampling.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            options.sampling.top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            options.sampling.top_k =
                static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--sampling-seed" && i + 1 < argc) {
            options.sampling.seed = std::stoull(argv[++i], nullptr, 0);
        } else if (arg == "--repetition-penalty" && i + 1 < argc) {
            options.sampling.repetition_penalty = std::stof(argv[++i]);
        } else if (arg == "--frequency-penalty" && i + 1 < argc) {
            options.sampling.frequency_penalty = std::stof(argv[++i]);
        } else if (arg == "--presence-penalty" && i + 1 < argc) {
            options.sampling.presence_penalty = std::stof(argv[++i]);
        } else if (arg == "--stop-token-id" && i + 1 < argc) {
            options.stop_token_ids.push_back(
                static_cast<std::int32_t>(std::stol(argv[++i])));
        } else if (arg == "--enable-logit-diagnostics") {
            enable_logit_diagnostics = true;
            options.retain_last_logits_for_diagnostics = true;
        } else {
            std::cerr << "usage: q38-runtime [--manifest FILE] [--stage0-index FILE "
                         "--stage1-index FILE] [--ple-layout FILE] [--chunk N] "
                         "[--identity FILE] [--sampling greedy|top-k-top-p] "
                         "[--snapshot-journal FILE] [--stop-token-id N] "
                         "[--enable-logit-diagnostics] "
                         "[sampling options]\n";
            return 2;
        }
    }

    try {
        if (!identity_path.empty())
            options.identity = q38::load_session_identity(identity_path);
        if (!manifest_path.empty()) {
            const auto manifest = q38::load_artifact_manifest(manifest_path);
            options.hidden_width = manifest.boundary_hidden_width;
            options.vocab_size = manifest.vocab_size;
            options.context_limit = manifest.context_limit;
        }
        if (stage0_index_path.empty() != stage1_index_path.empty())
            throw std::runtime_error("both stage tensor indexes are required");
        if (!stage0_index_path.empty()) {
            const auto stage0 = q38::load_tensor_index(stage0_index_path);
            const auto stage1 = q38::load_tensor_index(stage1_index_path);
            q38::validate_tensor_index_pair(stage0, stage1);
            (void)q38::build_qwen38_stage_plan(stage0);
            (void)q38::build_qwen38_stage_plan(stage1);
        }
        std::shared_ptr<q38::PleStore> ple_store;
        std::unique_ptr<q38::PleHashState> ple_hash;
        q38::PleHashConfigV1 ple_hash_config;
        if (!ple_layout_path.empty()) {
            auto layout = q38::load_ple_layout(ple_layout_path);
            ple_hash_config = layout.hash;
            ple_hash = std::make_unique<q38::PleHashState>(layout.hash);
            ple_store = std::make_shared<q38::PleStore>(
                std::move(layout), ple_cache_mib << 20u);
        }
        std::unique_ptr<q38::DualStageExecutor> executor;
        if (ple_store) {
            auto stage0 = std::make_unique<q38::PleStage0Backend>(
                std::make_unique<q38::MockStageBackend>(
                    q38::Stage::kStage0, options.hidden_width,
                    options.vocab_size),
                ple_store, ple_hash_config);
            auto stage1 = std::make_unique<q38::MockStageBackend>(
                q38::Stage::kStage1, options.hidden_width, options.vocab_size);
            executor = std::make_unique<q38::DualStageExecutor>(
                options, std::move(stage0), std::move(stage1));
        } else {
            executor = q38::make_mock_executor(options);
        }
        std::cout << "{\"type\":\"ready\",\"backend\":\"mock\","
                     "\"contract_version\":1,\"rpc_version\":1,"
                  << "\"logit_diagnostics_enabled\":"
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
                print_state(*executor);
                continue;
            }
            if (command == "stats") {
                const auto stats = executor->stats();
                std::cout << "{\"type\":\"stats\",\"transactions\":"
                          << stats.transactions
                          << ",\"evaluated_tokens\":"
                          << stats.evaluated_tokens
                          << ",\"published_tokens\":"
                          << stats.published_tokens
                          << ",\"drafted_tokens\":" << stats.drafted_tokens
                          << ",\"accepted_draft_tokens\":"
                          << stats.accepted_draft_tokens
                          << ",\"rejected_draft_tokens\":"
                          << stats.rejected_draft_tokens
                          << ",\"maximum_draft_width\":"
                          << stats.maximum_draft_width
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
                continue;
            }
            if (command == "append") {
                std::vector<std::int32_t> tokens;
                std::int64_t value = 0;
                while (input >> value) tokens.push_back(static_cast<std::int32_t>(value));
                executor->append(tokens);
                std::cout << "{\"type\":\"append\",\"count\":"
                          << tokens.size() << "}\n";
                continue;
            }
            if (command == "ple") {
                if (!ple_store || !ple_hash)
                    throw std::runtime_error("ple command requires --ple-layout");
                std::vector<std::int32_t> tokens;
                std::int64_t value = 0;
                while (input >> value)
                    tokens.push_back(static_cast<std::int32_t>(value));
                const auto rows = ple_hash->rows(tokens);
                const auto bytes = ple_store->read_rows(rows);
                const auto stats = ple_store->cache_stats();
                std::uint64_t digest = UINT64_C(1469598103934665603);
                for (const auto byte : bytes) {
                    digest ^= byte;
                    digest *= UINT64_C(1099511628211);
                }
                std::cout << "{\"type\":\"ple\",\"tokens\":" << tokens.size()
                          << ",\"rows\":" << rows.size()
                          << ",\"bytes\":" << bytes.size()
                          << ",\"digest\":" << digest
                          << ",\"cache_hits\":" << stats.hits
                          << ",\"cache_misses\":" << stats.misses
                          << ",\"cache_evictions\":" << stats.evictions
                          << ",\"physical_read_bytes\":"
                          << stats.physical_read_bytes << "}\n";
                continue;
            }
            if (command == "seed") {
                const auto token = executor->seed_decode();
                std::cout << "{\"type\":\"tokens\",\"tokens\":[" << token
                          << "]}\n";
                continue;
            }
            if (command == "decode") {
                std::uint32_t steps = 1;
                input >> steps;
                for (std::uint32_t i = 0; i < steps; ++i) {
                    const auto result = executor->decode_one();
                    std::cout << "{\"type\":\"tokens\",\"mode\":\"plain\","
                                 "\"tokens\":";
                    print_tokens(result.published_tokens);
                    std::cout << ",\"epoch\":" << result.commit.epoch << "}\n";
                }
                continue;
            }
            if (command == "spec") {
                std::uint32_t steps = 1;
                std::uint32_t max_draft = 3;
                input >> steps >> max_draft;
                for (std::uint32_t i = 0; i < steps; ++i) {
                    const auto result = executor->speculative_step(max_draft);
                    std::cout << "{\"type\":\"tokens\",\"mode\":\"mtp\","
                                 "\"tokens\":";
                    print_tokens(result.published_tokens);
                    std::cout << ",\"epoch\":" << result.commit.epoch << "}\n";
                }
                continue;
            }
            std::cout << "{\"type\":\"error\",\"message\":\"unknown command\"}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "q38-runtime: " << error.what() << '\n';
        return 1;
    }
}
