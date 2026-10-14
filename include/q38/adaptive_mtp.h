#ifndef Q38_ADAPTIVE_MTP_H
#define Q38_ADAPTIVE_MTP_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace q38 {

// Batch verification pays for every evaluated row, including rejected rows.
// Measure wall time per published token for each qualified graph shape instead
// of extrapolating the retained, early-exit verifier's cost per accepted row.
class BatchedMtpWidthPolicy {
public:
    std::uint32_t choose(std::uint32_t cap) const {
        if (!cap) return 0;
        // Two initial observations per arm amortize first-use graph capture.
        for (std::uint32_t pass = 0; pass < 2; ++pass)
            for (auto width : widths_)
                if (width <= cap && arms_[width].count <= pass) return width;
        std::uint32_t best = 0;
        for (auto width : widths_)
            if (width <= cap && arms_[width].cost < arms_[best].cost) best = width;
        // Revisit stale alternatives so a document change can recover from an
        // earlier bad width. This never changes candidate/token selection.
        if (observations_ % 8 == 0) {
            auto stale = best;
            for (auto width : widths_)
                if (width <= cap && arms_[width].last < arms_[stale].last) stale = width;
            return stale;
        }
        return best;
    }
    void observe(std::uint32_t width, std::uint32_t published,
                 std::uint64_t elapsed_ns, bool stopped = false) {
        if (width >= arms_.size() ||
            std::find(widths_.begin(), widths_.end(), width) == widths_.end() ||
            !published || published > width + 1 || !elapsed_ns || stopped) return;
        auto& arm = arms_[width];
        // The second sample replaces cold capture timing; subsequent samples
        // track changing acceptance and actual batch costs together.
        const double alpha = arm.count < 2 ? 1.0 : 0.25;
        // Throughput is total publications / total time, not the mean of
        // each transaction's time/publications ratio. The latter overweights
        // rejected batches and can prefer decode despite better MTP throughput.
        arm.elapsed += alpha * (static_cast<double>(elapsed_ns) - arm.elapsed);
        arm.published += alpha * (static_cast<double>(published) - arm.published);
        arm.cost = arm.elapsed / arm.published;
        ++arm.count;
        arm.last = ++observations_;
    }
    void reset() { *this = BatchedMtpWidthPolicy{}; }
    std::uint64_t observations() const { return observations_; }
private:
    struct Arm {
        double cost = 0, elapsed = 0, published = 0;
        std::uint64_t count = 0, last = 0;
    };
    static constexpr std::array<std::uint32_t, 5> widths_{{2, 0, 1, 4, 6}};
    std::array<Arm, 7> arms_{};
    std::uint64_t observations_ = 0;
};

// Performance-only state. Updates happen after commit, never affect token
// selection, and are discarded on session reset. A request's width is a cap.
class AdaptiveMtpPolicy {
public:
    std::uint32_t choose(std::uint32_t cap) const {
        cap = std::min<std::uint32_t>(cap, 64);
        if (cap == 0 || observations_ == 0) return std::min(cap, 2u);
        std::uint32_t best = 1;
        double best_cost = std::numeric_limits<double>::infinity();
        for (std::uint32_t width = 1; width <= cap; ++width) {
            double survival = 1.0;
            double published = 1.0;
            for (std::uint32_t depth = 0; depth < width; ++depth) {
                // A small prior prevents a single rejection from permanently
                // suppressing a depth. Unverified suffixes are not failures.
                survival *= (successes_[depth] + 1.0) /
                            (attempts_[depth] + 2.0);
                published += survival;
            }
            const double cost =
                (fixed_ns_ + draft_row_ns_ * width +
                 target_row_ns_ * published) / published;
            if (cost < best_cost) {
                best = width;
                best_cost = cost;
            }
        }
        // Bounded exploration lets the controller recover when the workload
        // changes. Only a neighbouring width is probed, never an unbounded jump.
        if (observations_ % 8 == 0) {
            if ((observations_ / 8) % 2 && best < cap) return best + 1;
            if (best > 1) return best - 1;
        }
        return best;
    }

    void observe(std::uint32_t width, std::uint32_t accepted,
                 bool stopped, std::uint64_t elapsed_ns,
                 std::uint64_t draft_ns, std::uint64_t target_ns) {
        if (width == 0 || width > 64 || accepted > width || elapsed_ns == 0)
            return;
        // Decay the acceptance history so old documents do not dominate.
        for (std::uint32_t i = 0; i < 64; ++i) {
            successes_[i] *= 0.95;
            attempts_[i] *= 0.95;
        }
        for (std::uint32_t i = 0; i < accepted; ++i) {
            ++successes_[i];
            ++attempts_[i];
        }
        if (!stopped && accepted < width) ++attempts_[accepted];
        // Normalize measured draft/target cost to the transaction wall time:
        // their timers overlap in the pipeline and must not be added twice.
        const double sum = static_cast<double>(draft_ns) + target_ns;
        const double scale = sum > elapsed_ns ? elapsed_ns / sum : 1.0;
        const double draft = draft_ns * scale;
        const double target = target_ns * scale;
        const double fixed = std::max(0.0, elapsed_ns - draft - target);
        const double alpha = observations_ == 0 ? 1.0 : 0.2;
        auto update = [alpha](double& old, double value) {
            old += alpha * (value - old);
        };
        update(draft_row_ns_, draft / width);
        update(target_row_ns_, target / (accepted + 1));
        update(fixed_ns_, fixed);
        ++observations_;
    }

    void reset() { *this = AdaptiveMtpPolicy{}; }
    std::uint64_t observations() const { return observations_; }

private:
    std::array<double, 64> successes_{};
    std::array<double, 64> attempts_{};
    double draft_row_ns_ = 0;
    double target_row_ns_ = 0;
    double fixed_ns_ = 0;
    std::uint64_t observations_ = 0;
};

}  // namespace q38
#endif
