#ifndef Q38_PAIRED_RANK_WORKER_H
#define Q38_PAIRED_RANK_WORKER_H

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace q38 {

// One serialized owner calls run(); rank 0 stays on that owner and rank 1
// reuses a worker. run() joins both logical tasks before captured references
// may disappear. abort_peer must be noexcept and must unblock rank barriers.
class PairedRankWorker {
public:
    explicit PairedRankWorker(std::function<void()> abort_peer)
        : abort_peer_(std::move(abort_peer)), worker_([this] { loop(); }) {}
    ~PairedRankWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        worker_.join();
    }
    PairedRankWorker(const PairedRankWorker&) = delete;
    PairedRankWorker& operator=(const PairedRankWorker&) = delete;

    template <class F> void run(F action) {
        std::exception_ptr local_error;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_ || stopping_)
                throw std::logic_error("paired rank worker requires one active owner");
            peer_error_ = nullptr;
            task_ = [&action] { action(1); };
            active_ = true;
            done_ = false;
        }
        ready_.notify_one();
        try { action(0); }
        catch (...) {
            local_error = std::current_exception();
            abort_peer_();
        }
        std::exception_ptr peer_error;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            finished_.wait(lock, [this] { return done_; });
            peer_error = peer_error_;
            active_ = false;
        }
        if (local_error) std::rethrow_exception(local_error);
        if (peer_error) std::rethrow_exception(peer_error);
    }

private:
    void loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || bool(task_); });
                if (stopping_ && !task_) return;
                task = std::move(task_);
                // A moved-from std::function is valid but can remain callable
                // (notably libc++ small-buffer targets). Consume it explicitly.
                task_ = {};
            }
            std::exception_ptr error;
            try { task(); }
            catch (...) {
                error = std::current_exception();
                abort_peer_();
            }
            // Destroy the reference-bearing callback before notifying owner.
            task = {};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                peer_error_ = error;
                done_ = true;
            }
            finished_.notify_one();
        }
    }
    std::function<void()> abort_peer_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::function<void()> task_;
    std::exception_ptr peer_error_;
    bool stopping_ = false, active_ = false, done_ = false;
    std::thread worker_;
};

} // namespace q38
#endif
