#include "q38/cuda_backend.h"
#include "q38/cuda_te.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "usage: " << argv[0]
                  << " r0s0 r0s1 r1s0 r1s1 ple identity session_hash [steps] [prompt-token-file] [logit-prefix] [pp]\n";
        return 2;
    }
    const int steps = argc > 8 ? std::stoi(argv[8]) : 64;
    const bool pp = argc > 11 && std::string(argv[11]) == "pp";
    const std::string logit_prefix = argc > 10 ? argv[10] : "";
    const bool diagnostics = !logit_prefix.empty() && logit_prefix != "-";
    if (steps < 1) throw std::invalid_argument("steps must be positive");
    q38::ExecutorOptions options;
    options.retain_last_logits_for_diagnostics = diagnostics;
    options.session_hash = std::stoull(argv[7], nullptr, 0);
    options.context_limit = 262144;
    options.append_chunk_tokens = 512;
    options.identity = q38::load_session_identity(argv[6]);
    options.sampling.mode = q38::SamplingModeV1::kGreedy;
    options.stop_token_ids = {248044};
    std::cerr << std::unitbuf;
    std::cerr << "te-decode-smoke: constructing ranks\n";
    q38::te_bridge_reset();
    auto rank0 = q38::make_cuda_executor(
        argv[1], argv[2], argv[5], options, 0, pp ? 1 : 0, 8ull << 30,
        q38::PleIoModeV1::kIoUringDirect, 64, false, false, pp ? 1 : 2, 0);
    auto rank1 = pp ? std::unique_ptr<q38::DualStageExecutor>{} : q38::make_cuda_executor(
        argv[3], argv[4], argv[5], options, 1, 1, 8ull << 30,
        q38::PleIoModeV1::kIoUringDirect, 64, false, false, 2, 1);


    std::vector<std::int32_t> prompt(1, 42);
    if (argc > 9 && std::string(argv[9]) != "-") {
        std::ifstream input(argv[9]);
        if (!input) throw std::runtime_error("cannot read prompt tokens");
        prompt.clear();
        std::int32_t token;
        while (input >> token) prompt.push_back(token);
        if (prompt.empty()) throw std::runtime_error("empty prompt");
    }
    auto capture = [&](const std::string& name) {
        if (!diagnostics) return;
        const auto a = rank0->diagnostic_logit_snapshot();
        if (!a) throw std::runtime_error("missing rank0 logits");
        if (rank1) {
            const auto b = rank1->diagnostic_logit_snapshot();
            if (!b || a->logits_bf16 != b->logits_bf16 ||
                a->selected_token != b->selected_token ||
                a->target_frontier != b->target_frontier)
                throw std::runtime_error("TE rank logit/frontier mismatch");
        }
        std::ofstream file(logit_prefix + "." + name + ".bf16", std::ios::binary);
        file.write(reinterpret_cast<const char*>(a->logits_bf16.data()),
                   a->logits_bf16.size() * sizeof(std::uint16_t));
        if (!file) throw std::runtime_error("cannot write diagnostic logits");
    };
    std::cerr << "te-decode-smoke: ranks ready, appending " << prompt.size() << "\n";
    std::exception_ptr e0, e1;
    auto append_both = [&](const std::vector<std::int32_t>& toks) {
        e0 = e1 = nullptr;
        std::thread t0([&] {
            try { rank0->append(toks); } catch (const std::exception& e) {
                std::cerr << "rank0 append exception: " << e.what() << "\n";
                q38::te_bridge_abort();
                e0 = std::current_exception();
            }
        });
        std::thread t1([&] {
            try { if (rank1) rank1->append(toks); } catch (const std::exception& e) {
                std::cerr << "rank1 append exception: " << e.what() << "\n";
                q38::te_bridge_abort();
                e1 = std::current_exception();
            }
        });
        t0.join(); t1.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
    };
    auto decode_both = [&]() -> std::int32_t {
        e0 = e1 = nullptr;
        q38::DecodeResult r0, r1;
        std::thread t0([&] {
            try { r0 = rank0->decode_one(); } catch (const std::exception& e) {
                std::cerr << "rank0 decode exception: " << e.what() << "\n";
                q38::te_bridge_abort();
                e0 = std::current_exception();
            }
        });
        std::thread t1([&] {
            try { if (rank1) r1 = rank1->decode_one(); } catch (const std::exception& e) {
                std::cerr << "rank1 decode exception: " << e.what() << "\n";
                q38::te_bridge_abort();
                e1 = std::current_exception();
            }
        });
        t0.join(); t1.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
        if (rank1 && (r0.published_tokens != r1.published_tokens ||
                      rank0->canonical_tokens() != rank1->canonical_tokens()))
            throw std::runtime_error("TE rank decode token mismatch");
        if (r0.published_tokens.empty())
            throw std::runtime_error("rank0 produced no token");
        return r0.published_tokens.back();
    };

    try {
        append_both(prompt);
        capture("append");
        std::cerr << "te-decode-smoke: append done, seeding decode\n";
        e0 = e1 = nullptr;
        std::int32_t seed0 = 0, seed1 = 0;
        std::thread ts0([&] {
            try { seed0 = rank0->seed_decode(); } catch (const std::exception& e) {
                std::cerr << "rank0 seed exception: " << e.what() << "\n";
                q38::te_bridge_abort();
                e0 = std::current_exception();
            }
        });
        std::thread ts1([&] {
            try { if (rank1) seed1 = rank1->seed_decode(); } catch (const std::exception& e) {
                std::cerr << "rank1 seed exception: " << e.what() << "\n";
                q38::te_bridge_abort();
                e1 = std::current_exception();
            }
        });
        ts0.join(); ts1.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
        if (rank1 && seed0 != seed1)
            throw std::runtime_error("TE seed mismatch: " + std::to_string(seed0) +
                                     " vs " + std::to_string(seed1));
        std::cerr << "te-decode-smoke: seed=" << seed0 << ", warmup\n";
        const bool ignore_eos = std::getenv("Q38_TE_SMOKE_IGNORE_EOS") != nullptr;
        bool stopped = !ignore_eos && seed0 == 248044;
        std::vector<std::int32_t> warmup;
        for (int i = 0; i < 4 && !stopped; ++i) {
            warmup.push_back(decode_both());
            capture("warmup" + std::to_string(i));
            stopped = !ignore_eos && warmup.back() == 248044;
            std::cerr << "te-decode-smoke: warmup " << (i + 1) << "\n";
        }
        const auto stats0 = q38::te_bridge_stats();
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::int32_t> out;
        out.reserve(steps);
        for (int i = 0; i < steps && !stopped; ++i) {
            out.push_back(decode_both());
            capture("step" + std::to_string(i));
            stopped = !ignore_eos && out.back() == 248044;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double sec =
            std::chrono::duration<double>(t1 - t0).count();
        const double tps = out.size() / sec;
        const auto stats1 = q38::te_bridge_stats();
        std::cout << "{\"schema\":\"Q38_TE_DECODE_SMOKE_V1\",\"steps\":" << out.size()
                  << ",\"requested_steps\":" << steps
                  << ",\"allgather_calls\":" << stats1.allgather_calls-stats0.allgather_calls
                  << ",\"allreduce_calls\":" << stats1.allreduce_calls-stats0.allreduce_calls
                  << ",\"collective_seconds\":" << (stats1.collective_ns-stats0.collective_ns)*1e-9
                  << ",\"stopped\":" << (stopped ? "true" : "false")
                  << ",\"wall_seconds\":" << sec
                  << ",\"tokens_per_second\":" << tps
                  << ",\"failures\":0,\"te_world_size\":" << (pp ? 1 : 2)
                  << ",\"prompt_tokens\":" << prompt.size()
                  << ",\"rank_token_parity\":true,\"diagnostics\":" << (diagnostics ? "true" : "false")
                  << ",\"seed\":" << seed0
                  << ",\"tokens\":[";
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << out[i];
        }
        std::cout << "],\"warmup\":[";
        for (std::size_t i = 0; i < warmup.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << warmup[i];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
