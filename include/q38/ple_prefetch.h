#ifndef Q38_PLE_PREFETCH_H
#define Q38_PLE_PREFETCH_H
#include "q38/ple.h"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace q38 {
// Two ready tiles bound host memory independently of the full transaction.
// Consumers may split or span tiles; cancellation joins the producer before reuse.
class PleAsyncReader {
public:
    explicit PleAsyncReader(std::shared_ptr<PleStore> value_store,
                            std::size_t tile_rows)
        : store_(std::move(value_store)), tile_rows_(tile_rows) {
        if (!store_ || tile_rows_ == 0 || tile_rows_ > 65536)
            throw std::invalid_argument("invalid PLE async store/tile");
        worker_ = std::thread([this] { run(); });
    }

    ~PleAsyncReader() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            cancel_requested_ = true;
        }
        ready_.notify_all();
        finished_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    PleAsyncReader(const PleAsyncReader&) = delete;
    PleAsyncReader& operator=(const PleAsyncReader&) = delete;

    void submit(std::shared_ptr<const std::vector<std::uint64_t>> rows) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!rows || rows->empty())
            throw std::invalid_argument("PLE async row set is empty");
        if (request_ready_ || working_ || result_ready_ || stopping_)
            throw std::logic_error("PLE async reader already has work");
        rows_ = std::move(rows);
        error_ = nullptr;
        total_rows_ = rows_->size();
        consumed_rows_ = 0;
        cancel_requested_ = false;
        request_ready_ = true;
        ready_.notify_one();
    }

    void take(std::size_t first_row, std::size_t row_count,
              std::uint8_t* output, std::size_t output_bytes,
              std::uint16_t* scale_output,
              std::size_t output_scales) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto row_bytes = store_->layout().row_stride_bytes;
        if (row_count == 0 || first_row != consumed_rows_ || first_row > total_rows_ ||
            row_count > total_rows_ - first_row || !output ||
            output_bytes != row_count * row_bytes || !scale_output ||
            output_scales != row_count)
            throw std::invalid_argument("PLE async take extent is invalid");
        // Consume an exact prefix, even when compute and I/O tiles differ.
        std::size_t copied = 0;
        while (copied < row_count) {
            const auto cursor = first_row + copied;
            finished_.wait(lock, [this, cursor] {
                return (!ready_tiles_.empty() &&
                        ready_tiles_.front().first_row == cursor) ||
                       result_ready_ || stopping_;
            });
            if (ready_tiles_.empty() || ready_tiles_.front().first_row != cursor) {
                if (error_) std::rethrow_exception(error_);
                throw std::runtime_error("PLE async reader stopped early");
            }
            auto& tile = ready_tiles_.front();
            const auto count = std::min(row_count - copied, tile.rows);
            std::copy_n(tile.bytes.data() + tile.consumed * row_bytes,
                        count * row_bytes, output + copied * row_bytes);
            std::copy_n(tile.scales.data() + tile.consumed, count,
                        scale_output + copied);
            copied += count;
            consumed_rows_ += count;
            tile.consumed += count;
            tile.first_row += count;
            tile.rows -= count;
            if (tile.rows == 0) {
                ready_tiles_.pop_front();
                ready_.notify_one();
            }
        }
        if (first_row + row_count == total_rows_) {
            finished_.wait(lock, [this] { return result_ready_ || stopping_; });
            const auto error = error_;
            result_ready_ = false;
            error_ = nullptr;
            if (error) std::rethrow_exception(error);
        }
    }

    void cancel() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!request_ready_ && !working_ && !result_ready_) return;
        cancel_requested_ = true;
        ready_.notify_all();
        finished_.notify_all();
        finished_.wait(lock, [this] {
            return result_ready_ ||
                   (!request_ready_ && !working_) || stopping_;
        });
        result_ready_ = false;
        error_ = nullptr;
        rows_.reset();
        ready_tiles_.clear();
        total_rows_ = 0;
        consumed_rows_ = 0;
        cancel_requested_ = false;
    }

private:
    struct PleTileResult {
        std::size_t first_row = 0;
        std::size_t rows = 0;
        std::size_t consumed = 0;
        std::vector<std::uint8_t> bytes;
        std::vector<std::uint16_t> scales;
    };

    void run() noexcept {
        for (;;) {
            std::shared_ptr<const std::vector<std::uint64_t>> rows;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,
                            [this] { return stopping_ || request_ready_; });
                if (stopping_ && !request_ready_) return;
                rows = std::move(rows_);
                request_ready_ = false;
                working_ = true;
            }
            std::exception_ptr error;
            try {
                if (!rows || rows->empty())
                    throw std::invalid_argument("PLE async row set vanished");
                const std::size_t kRowsPerTile = tile_rows_;
                for (std::size_t first = 0; first < rows->size();
                     first += kRowsPerTile) {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait(lock, [this] {
                            return ready_tiles_.size() < 2 ||
                                   cancel_requested_ || stopping_;
                        });
                        if (cancel_requested_ || stopping_) break;
                    }
                    const auto count =
                        std::min<std::size_t>(kRowsPerTile,
                                              rows->size() - first);
                    std::vector<std::uint64_t> tile(
                        rows->begin() + static_cast<std::ptrdiff_t>(first),
                        rows->begin() +
                            static_cast<std::ptrdiff_t>(first + count));
                    PleTileResult result;
                    result.first_row = first;
                    result.rows = count;
                    result.bytes = store_->read_rows(tile);
                    result.scales.resize(count);
                    store_->read_row_scales_into(
                        tile, result.scales.data(), result.scales.size());
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        ready_tiles_.push_back(std::move(result));
                        finished_.notify_all();
                    }
                }
            } catch (...) {
                error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                working_ = false;
                result_ready_ = true;
                error_ = error;
            }
            finished_.notify_one();
        }
    }

    std::shared_ptr<PleStore> store_;
    const std::size_t tile_rows_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::shared_ptr<const std::vector<std::uint64_t>> rows_;
    std::deque<PleTileResult> ready_tiles_;
    std::exception_ptr error_;
    std::size_t total_rows_ = 0;
    std::size_t consumed_rows_ = 0;
    bool request_ready_ = false;
    bool working_ = false;
    bool result_ready_ = false;
    bool cancel_requested_ = false;
    bool stopping_ = false;
};

}  // namespace q38
#endif
