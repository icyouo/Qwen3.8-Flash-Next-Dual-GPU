#ifndef Q38_PLE_H
#define Q38_PLE_H

#include "q38/contracts.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q38 {

constexpr std::uint32_t kPleHeads = 16;
constexpr std::uint32_t kPleHeadsPerNgram = 8;

struct PleHashConfigV1 {
    std::uint32_t unigram_vocab_size = 0;
    std::uint32_t eos_token_id = 0;
    std::array<std::uint64_t, 3> multipliers{};
    std::array<std::uint64_t, kPleHeads> head_vocab_sizes{};
    std::array<std::uint64_t, kPleHeads> head_offsets{};
};

struct PleFileV1 {
    std::uint32_t index = 0;
    std::string path;
    std::uint64_t file_bytes = 0;
    std::uint64_t payload_bytes = 0;
};

struct PlePartV1 {
    std::uint32_t logical_part = 0;
    std::uint32_t file_index = 0;
    std::uint64_t global_row_start = 0;
    std::uint64_t rows = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t payload_bytes = 0;
    std::uint32_t scale_file_index = 0;
    std::uint64_t scale_file_offset = 0;
    std::uint64_t scale_bytes = 0;
};

struct PleLayoutV1 {
    DType storage_dtype = DType::kInvalid;
    std::uint32_t alignment_bytes = 0;
    std::uint32_t row_stride_bytes = 0;
    std::uint32_t row_dimension = 0;
    std::uint64_t usable_rows = 0;
    std::uint64_t padded_rows = 0;
    PleHashConfigV1 hash{};
    std::vector<PleFileV1> files;
    std::vector<PlePartV1> parts;
};

struct PleCacheStatsV1 {
    std::uint64_t requested_rows = 0;
    std::uint64_t unique_page_requests = 0;
    std::uint64_t useful_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t physical_read_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t read_operations = 0;
    std::uint64_t read_batches = 0;
    std::uint64_t io_uring_submissions = 0;
    std::uint64_t io_uring_completions = 0;
    std::uint64_t direct_read_bytes = 0;
    std::uint64_t read_errors = 0;
    std::uint64_t maximum_queue_depth = 0;
    std::uint64_t read_latency_p50_ns = 0;
    std::uint64_t read_latency_p95_ns = 0;
    std::uint64_t read_latency_p99_ns = 0;
    std::uint64_t io_uring_enabled = 0;
    std::uint64_t direct_io_enabled = 0;
    std::uint64_t scale_resident_bytes = 0;
};

enum class PleIoModeV1 : std::uint32_t {
    kAuto = 0,
    kBuffered = 1,
    kIoUringDirect = 2,
};

struct PleStoreOptionsV1 {
    std::uint64_t cache_bytes = 0;
    std::uint32_t cache_page_bytes = 4096;
    PleIoModeV1 io_mode = PleIoModeV1::kAuto;
    std::uint32_t queue_depth = 64;
};

struct PleReadExtentV1 {
    std::uint32_t file_index = 0;
    std::uint64_t offset = 0;
    std::size_t output_offset = 0;
    std::size_t bytes = 0;
};

PleLayoutV1 load_ple_layout(const std::string& path,
                            bool verify_source_files = true);

class PleHashState {
public:
    explicit PleHashState(const PleHashConfigV1& config);
    void reset();
    std::vector<std::uint64_t> rows(
        const std::vector<std::int32_t>& token_ids);

private:
    PleHashConfigV1 config_;
    std::array<std::int64_t, 2> previous_{};
};

class PleStore {
public:
    explicit PleStore(PleLayoutV1 layout, std::uint64_t cache_bytes = 0,
                      std::uint32_t cache_page_bytes = 4096);
    PleStore(PleLayoutV1 layout, PleStoreOptionsV1 options);
    ~PleStore();

    PleStore(const PleStore&) = delete;
    PleStore& operator=(const PleStore&) = delete;

    const PleLayoutV1& layout() const { return layout_; }
    std::vector<std::uint8_t> read_row(std::uint64_t global_row) const;
    std::vector<std::uint8_t> read_rows(
        const std::vector<std::uint64_t>& global_rows) const;
    void read_rows_into(const std::vector<std::uint64_t>& global_rows,
                        std::uint8_t* output,
                        std::size_t output_bytes) const;
    void read_row_scales_into(const std::vector<std::uint64_t>& global_rows,
                              std::uint16_t* output,
                              std::size_t output_scales) const;
    PleCacheStatsV1 cache_stats() const;

private:
    struct OpenFile {
        std::uint32_t index = 0;
        int fd = -1;
        int direct_fd = -1;
    };

    struct Cache;
    struct ScalePart;
    struct IoUringProvider;

    const PlePartV1& find_part(std::uint64_t global_row) const;
    int file_descriptor(std::uint32_t file_index) const;
    std::uint64_t file_size(std::uint32_t file_index) const;
    void read_exact(std::uint32_t file_index, std::uint64_t offset,
                    std::uint8_t* output, std::size_t size) const;
    void read_extents(const std::vector<PleReadExtentV1>& extents,
                      std::uint8_t* output, std::size_t output_bytes) const;

    PleLayoutV1 layout_;
    std::vector<OpenFile> files_;
    std::vector<ScalePart> scale_parts_;
    std::unique_ptr<Cache> cache_;
    std::unique_ptr<IoUringProvider> io_uring_;
    std::uint64_t scale_resident_bytes_ = 0;
};

}  // namespace q38

#endif
