#include "q38/cuda_backend.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "usage: " << argv[0]
                  << " r0s0 r0s1 r1s0 r1s1 ple identity session_hash\n";
        return 2;
    }
    q38::ExecutorOptions options;
    options.session_hash = std::stoull(argv[7], nullptr, 0);
    options.context_limit = 262144;
    options.append_chunk_tokens = 512;
    options.identity = q38::load_session_identity(argv[6]);
    options.sampling.mode = q38::SamplingModeV1::kGreedy;
    options.stop_token_ids = {248044};
    try {
        std::cerr << "loading rank0 TE on GPU0...\n";
        auto rank0 = q38::make_cuda_executor(
            argv[1], argv[2], argv[5], options, 0, 0, 8ull << 30,
            q38::PleIoModeV1::kIoUringDirect, 64, false, false, 2, 0);
        std::cerr << "rank0 OK\n";
        std::cerr << "loading rank1 TE on GPU1...\n";
        auto rank1 = q38::make_cuda_executor(
            argv[3], argv[4], argv[5], options, 1, 1, 8ull << 30,
            q38::PleIoModeV1::kIoUringDirect, 64, false, false, 2, 1);
        std::cerr << "rank1 OK\n";
        std::cout << "{\"te_bind\":\"PASS\"}\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
