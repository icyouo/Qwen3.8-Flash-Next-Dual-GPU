#include "q38/paired_rank_worker.h"
#include <atomic>
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        std::atomic<unsigned> aborts{0};
        q38::PairedRankWorker worker([&] { ++aborts; });
        std::thread::id owner = std::this_thread::get_id(), peer;
        unsigned count[2] = {};
        for (unsigned generation = 0; generation < 10000; ++generation) {
            worker.run([&](int rank) {
                if (rank == 0) require(std::this_thread::get_id() == owner, "owner changed");
                else if (generation == 0) peer = std::this_thread::get_id();
                else require(std::this_thread::get_id() == peer, "peer was recreated");
                require(count[rank] == generation, "generation mismatch");
                ++count[rank];
            });
        }
        require(peer != owner && count[0] == 10000 && count[1] == 10000, "work incomplete");
        for (int failing_rank = 0; failing_rank < 2; ++failing_rank) {
            bool caught = false;
            const unsigned before = aborts;
            try {
                worker.run([&](int rank) {
                    if (rank == failing_rank) throw std::runtime_error("rank error");
                    // Simulate a peer parked in a collective until abort.
                    while (aborts.load() == before) std::this_thread::yield();
                });
            } catch (const std::runtime_error&) { caught = true; }
            require(caught && aborts == before + 1, "failed peer not unblocked");
        }
        bool nonstandard = false;
        try { worker.run([](int rank) { if (rank) throw 17; }); }
        catch (int value) { nonstandard = value == 17; }
        require(nonstandard, "nonstandard exception lost");
        unsigned completed = 0;
        worker.run([&](int rank) { if (rank == 1) ++completed; });
        require(completed == 1, "worker did not join task");
        for (int i = 0; i < 100; ++i) {
            q38::PairedRankWorker temporary([] {});
            temporary.run([](int) {});
        }
        std::cout << "PASS persistent rank worker: generations, identity, abort, exceptions, teardown\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
