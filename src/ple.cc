#include "q38/ple.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/syscall.h>
#endif

namespace q38 {

namespace {

std::uint64_t parse_u64(const std::string& field, const std::string& value) {
    std::size_t used = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &used, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("PLE layout " + field + " is not an integer");
    }
    if (used != value.size())
        throw std::runtime_error("PLE layout " + field + " has trailing bytes");
    return parsed;
}

std::uint32_t parse_u32(const std::string& field, const std::string& value) {
    const auto parsed = parse_u64(field, value);
    if (parsed > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("PLE layout " + field + " exceeds uint32");
    return static_cast<std::uint32_t>(parsed);
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> fields;
    std::istringstream input(value);
    std::string field;
    while (std::getline(input, field, separator)) fields.push_back(field);
    return fields;
}

template <std::size_t Size>
std::array<std::uint64_t, Size> parse_array(const std::string& field,
                                            const std::string& value) {
    const auto fields = split(value, ',');
    if (fields.size() != Size)
        throw std::runtime_error("PLE layout " + field + " has wrong length");
    std::array<std::uint64_t, Size> result{};
    for (std::size_t i = 0; i < Size; ++i)
        result[i] = parse_u64(field, fields[i]);
    return result;
}

DType parse_dtype(const std::string& value) {
    if (value == "bf16") return DType::kBFloat16;
    if (value == "fp8_e4m3") return DType::kFp8E4M3;
    throw std::runtime_error("PLE layout has unsupported storage dtype");
}

void validate_layout(PleLayoutV1* layout, bool verify_source_files) {
    const auto element_bytes = dtype_bytes(layout->storage_dtype);
    if (element_bytes == 0 || layout->alignment_bytes == 0 ||
        layout->row_stride_bytes == 0 || layout->row_dimension == 0 ||
        layout->usable_rows == 0 || layout->padded_rows < layout->usable_rows)
        throw std::runtime_error("PLE layout has invalid dimensions");
    if (static_cast<std::uint64_t>(layout->row_dimension) * element_bytes !=
        layout->row_stride_bytes)
        throw std::runtime_error("PLE row stride does not match dtype and dimension");
    if (layout->files.empty() || layout->parts.empty())
        throw std::runtime_error("PLE layout has no files or logical parts");
    if (layout->hash.unigram_vocab_size == 0 ||
        layout->hash.eos_token_id >= layout->hash.unigram_vocab_size)
        throw std::runtime_error("PLE hash token configuration is invalid");

    std::uint64_t expected_offset = 0;
    for (std::size_t head = 0; head < kPleHeads; ++head) {
        if (layout->hash.head_vocab_sizes[head] == 0 ||
            layout->hash.head_offsets[head] != expected_offset)
            throw std::runtime_error("PLE hash head ranges are not contiguous");
        expected_offset += layout->hash.head_vocab_sizes[head];
    }
    if (expected_offset != layout->usable_rows)
        throw std::runtime_error("PLE hash head rows do not equal usable rows");

    std::sort(layout->files.begin(), layout->files.end(),
              [](const PleFileV1& a, const PleFileV1& b) {
                  return a.index < b.index;
              });
    for (std::size_t i = 0; i < layout->files.size(); ++i) {
        const auto& file = layout->files[i];
        if (file.index != i || file.path.empty() || file.file_bytes == 0 ||
            file.payload_bytes > file.file_bytes)
            throw std::runtime_error("PLE physical file table is invalid");
        if (verify_source_files) {
            std::error_code error;
            const auto size = std::filesystem::file_size(file.path, error);
            if (error || size != file.file_bytes)
                throw std::runtime_error("PLE physical file size mismatch: " +
                                         file.path);
        }
    }

    std::sort(layout->parts.begin(), layout->parts.end(),
              [](const PlePartV1& a, const PlePartV1& b) {
                  return a.global_row_start < b.global_row_start;
              });
    std::uint64_t next_row = 0;
    for (std::size_t i = 0; i < layout->parts.size(); ++i) {
        const auto& part = layout->parts[i];
        if (part.logical_part != i || part.file_index >= layout->files.size() ||
            part.global_row_start != next_row || part.rows == 0 ||
            part.payload_bytes != part.rows * layout->row_stride_bytes)
            throw std::runtime_error("PLE logical part table is invalid");
        const auto& file = layout->files[part.file_index];
        if (part.file_offset > file.file_bytes ||
            part.payload_bytes > file.file_bytes - part.file_offset)
            throw std::runtime_error("PLE logical part exceeds its physical file");
        if (layout->storage_dtype == DType::kFp8E4M3) {
            if (part.scale_file_index >= layout->files.size() ||
                part.scale_bytes != part.rows * sizeof(std::uint16_t))
                throw std::runtime_error("PLE row-scale table is invalid");
            const auto& scale_file = layout->files[part.scale_file_index];
            if (part.scale_file_offset > scale_file.file_bytes ||
                part.scale_bytes >
                    scale_file.file_bytes - part.scale_file_offset)
                throw std::runtime_error("PLE row-scale extent exceeds its file");
        } else if (part.scale_bytes != 0) {
            throw std::runtime_error("non-FP8 PLE part has row scales");
        }
        next_row += part.rows;
    }
    if (next_row != layout->padded_rows)
        throw std::runtime_error("PLE logical parts do not cover padded rows");
}

}  // namespace

PleLayoutV1 load_ple_layout(const std::string& path,
                            bool verify_source_files) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open PLE layout: " + path);
    std::string line;
    if (!std::getline(input, line) || line != "Q38_PLE_LAYOUT_V1")
        throw std::runtime_error("PLE layout has invalid magic/version");

    PleLayoutV1 layout;
    std::set<std::string> headers;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto fields = split(line, '\t');
        if (fields.size() > 1) {
            if (fields[0] == "file" && fields.size() == 5) {
                PleFileV1 file;
                file.index = parse_u32("file index", fields[1]);
                file.path = fields[2];
                file.file_bytes = parse_u64("file bytes", fields[3]);
                file.payload_bytes = parse_u64("file payload", fields[4]);
                layout.files.push_back(std::move(file));
                continue;
            }
            if (fields[0] == "part" &&
                (fields.size() == 7 || fields.size() == 10)) {
                PlePartV1 part;
                part.logical_part = parse_u32("part index", fields[1]);
                part.file_index = parse_u32("part file", fields[2]);
                part.global_row_start = parse_u64("part row start", fields[3]);
                part.rows = parse_u64("part rows", fields[4]);
                part.file_offset = parse_u64("part file offset", fields[5]);
                part.payload_bytes = parse_u64("part payload", fields[6]);
                if (fields.size() == 10) {
                    part.scale_file_index =
                        parse_u32("part scale file", fields[7]);
                    part.scale_file_offset =
                        parse_u64("part scale offset", fields[8]);
                    part.scale_bytes =
                        parse_u64("part scale bytes", fields[9]);
                }
                layout.parts.push_back(part);
                continue;
            }
            throw std::runtime_error("PLE layout has invalid tab record");
        }

        const auto equals = line.find('=');
        if (equals == std::string::npos)
            throw std::runtime_error("PLE layout line has no '='");
        const auto key = line.substr(0, equals);
        const auto value = line.substr(equals + 1);
        if (!headers.insert(key).second)
            throw std::runtime_error("PLE layout has duplicate key " + key);
        if (key == "dtype") layout.storage_dtype = parse_dtype(value);
        else if (key == "alignment") layout.alignment_bytes = parse_u32(key, value);
        else if (key == "row_stride") layout.row_stride_bytes = parse_u32(key, value);
        else if (key == "row_dimension") layout.row_dimension = parse_u32(key, value);
        else if (key == "usable_rows") layout.usable_rows = parse_u64(key, value);
        else if (key == "padded_rows") layout.padded_rows = parse_u64(key, value);
        else if (key == "unigram_vocab")
            layout.hash.unigram_vocab_size = parse_u32(key, value);
        else if (key == "eos_token") layout.hash.eos_token_id = parse_u32(key, value);
        else if (key == "multipliers")
            layout.hash.multipliers = parse_array<3>(key, value);
        else if (key == "head_vocab_sizes")
            layout.hash.head_vocab_sizes = parse_array<kPleHeads>(key, value);
        else if (key == "head_offsets")
            layout.hash.head_offsets = parse_array<kPleHeads>(key, value);
        else throw std::runtime_error("PLE layout has unknown key " + key);
    }
    if (headers.size() != 11)
        throw std::runtime_error("PLE layout is missing required headers");
    validate_layout(&layout, verify_source_files);
    return layout;
}

PleHashState::PleHashState(const PleHashConfigV1& config) : config_(config) {
    reset();
}

void PleHashState::reset() {
    previous_[0] = config_.eos_token_id;
    previous_[1] = config_.eos_token_id;
}

std::vector<std::uint64_t> PleHashState::rows(
    const std::vector<std::int32_t>& token_ids) {
    for (const auto token : token_ids) {
        if (token < 0 || static_cast<std::uint32_t>(token) >=
                             config_.unigram_vocab_size)
            throw std::runtime_error("PLE hash token is outside vocabulary");
    }
    std::vector<std::uint64_t> result(token_ids.size() * kPleHeads);
    auto older = previous_[0];
    auto newer = previous_[1];
    const auto eos = static_cast<std::int64_t>(config_.eos_token_id);
    for (std::size_t token_index = 0; token_index < token_ids.size(); ++token_index) {
        const auto current = static_cast<std::uint64_t>(token_ids[token_index]);
        const auto previous1 = static_cast<std::uint64_t>(newer);
        const auto previous2 = static_cast<std::uint64_t>(newer == eos ? eos : older);
        const auto bigram = current * config_.multipliers[0] ^
                            previous1 * config_.multipliers[1];
        const auto trigram = bigram ^ previous2 * config_.multipliers[2];
        for (std::uint32_t head = 0; head < kPleHeadsPerNgram; ++head)
            result[token_index * kPleHeads + head] =
                bigram % config_.head_vocab_sizes[head] +
                config_.head_offsets[head];
        for (std::uint32_t local = 0; local < kPleHeadsPerNgram; ++local) {
            const auto head = kPleHeadsPerNgram + local;
            result[token_index * kPleHeads + head] =
                trigram % config_.head_vocab_sizes[head] +
                config_.head_offsets[head];
        }
        older = newer;
        newer = token_ids[token_index];
    }
    previous_[0] = older;
    previous_[1] = newer;
    return result;
}

struct PleStore::Cache {
    struct Entry {
        std::vector<std::uint8_t> bytes;
        std::list<std::uint64_t>::iterator lru;
    };

    Cache(std::uint64_t capacity, std::uint32_t page)
        : capacity_bytes(capacity), page_bytes(page),
          max_pages(static_cast<std::size_t>(capacity / page)) {}

    std::uint64_t capacity_bytes;
    std::uint32_t page_bytes;
    std::size_t max_pages;
    mutable std::mutex mutex;
    mutable std::unordered_map<std::uint64_t, Entry> entries;
    mutable std::list<std::uint64_t> lru;
    mutable PleCacheStatsV1 stats;
    mutable std::deque<std::uint64_t> read_latencies;
};

struct PleStore::ScalePart {
    std::vector<std::uint16_t> values;
};

struct PleStore::IoUringProvider {
    struct ReadRequest {
        int fd = -1;
        std::uint64_t offset = 0;
        std::uint8_t* output = nullptr;
        std::size_t expected_bytes = 0;
    };

#ifdef __linux__
    IoUringProvider(std::uint32_t queue_depth, std::uint32_t page_bytes)
        : queue_depth_(queue_depth), page_bytes_(page_bytes) {
        if (queue_depth_ == 0 || page_bytes_ != 4096)
            throw std::invalid_argument("invalid PLE io_uring dimensions");
        io_uring_params parameters{};
        ring_fd_ = static_cast<int>(
            ::syscall(__NR_io_uring_setup, queue_depth_, &parameters));
        if (ring_fd_ < 0)
            throw std::runtime_error("io_uring_setup failed: " +
                                     std::string(std::strerror(errno)));
        try {
            map_rings(parameters);
            allocate_buffers();
        } catch (...) {
            reset();
            throw;
        }
    }

    ~IoUringProvider() { reset(); }

    IoUringProvider(const IoUringProvider&) = delete;
    IoUringProvider& operator=(const IoUringProvider&) = delete;

    std::uint32_t queue_depth() const { return queue_depth_; }

    std::uint64_t read(const std::vector<ReadRequest>& requests) {
        if (requests.empty() || requests.size() > queue_depth_)
            throw std::invalid_argument("PLE io_uring batch size is invalid");
        const auto count = static_cast<std::uint32_t>(requests.size());
        const auto head = load(sq_head_);
        auto tail = load(sq_tail_);
        if (tail - head + count > *sq_entries_)
            throw std::runtime_error("PLE io_uring submission queue is full");
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto slot = tail & *sq_ring_mask_;
            auto& entry = sqes_[slot];
            std::memset(&entry, 0, sizeof(entry));
            entry.opcode = IORING_OP_READ_FIXED;
            entry.fd = requests[index].fd;
            entry.off = requests[index].offset;
            entry.addr = reinterpret_cast<std::uint64_t>(buffers_[index]);
            entry.len = page_bytes_;
            entry.buf_index = static_cast<std::uint16_t>(index);
            entry.user_data = index;
            sq_array_[slot] = slot;
            ++tail;
        }
        store(sq_tail_, tail);
        std::uint32_t submitted = 0;
        while (submitted < count) {
            const auto result = static_cast<int>(::syscall(
                __NR_io_uring_enter, ring_fd_, count - submitted, count,
                IORING_ENTER_GETEVENTS, nullptr, 0));
            if (result < 0 && errno == EINTR) continue;
            if (result < 0)
                throw std::runtime_error("io_uring_enter failed: " +
                                         std::string(std::strerror(errno)));
            submitted += static_cast<std::uint32_t>(result);
        }

        while (load(cq_tail_) - load(cq_head_) < count) {
            const auto available = load(cq_tail_) - load(cq_head_);
            const auto result = static_cast<int>(::syscall(
                __NR_io_uring_enter, ring_fd_, 0, count - available,
                IORING_ENTER_GETEVENTS, nullptr, 0));
            if (result < 0 && errno == EINTR) continue;
            if (result < 0)
                throw std::runtime_error("io_uring completion wait failed: " +
                                         std::string(std::strerror(errno)));
        }
        auto completion_head = load(cq_head_);
        std::uint64_t physical_bytes = 0;
        for (std::uint32_t done = 0; done < count; ++done) {
            const auto& completion =
                cqes_[completion_head & *cq_ring_mask_];
            const auto request_index =
                static_cast<std::size_t>(completion.user_data);
            if (request_index >= requests.size() || completion.res < 0)
                throw std::runtime_error(
                    completion.res < 0
                        ? "PLE io_uring read failed: " +
                              std::string(std::strerror(-completion.res))
                        : "PLE io_uring completion identity is invalid");
            const auto bytes = static_cast<std::size_t>(completion.res);
            if (bytes < requests[request_index].expected_bytes ||
                bytes > page_bytes_)
                throw std::runtime_error("PLE io_uring direct read was short");
            std::copy_n(static_cast<std::uint8_t*>(buffers_[request_index]),
                        bytes, requests[request_index].output);
            physical_bytes += bytes;
            ++completion_head;
        }
        store(cq_head_, completion_head);
        return physical_bytes;
    }

private:
    static std::uint32_t load(const std::uint32_t* value) {
        return __atomic_load_n(value, __ATOMIC_ACQUIRE);
    }
    static void store(std::uint32_t* target, std::uint32_t value) {
        __atomic_store_n(target, value, __ATOMIC_RELEASE);
    }

    void map_rings(const io_uring_params& parameters) {
        sq_ring_bytes_ =
            parameters.sq_off.array + parameters.sq_entries * sizeof(unsigned);
        cq_ring_bytes_ =
            parameters.cq_off.cqes + parameters.cq_entries * sizeof(io_uring_cqe);
        if (parameters.features & IORING_FEAT_SINGLE_MMAP) {
            const auto bytes = std::max(sq_ring_bytes_, cq_ring_bytes_);
            sq_ring_ = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                              MAP_SHARED, ring_fd_, IORING_OFF_SQ_RING);
            if (sq_ring_ == MAP_FAILED) {
                sq_ring_ = nullptr;
                throw std::runtime_error("mmap io_uring rings failed: " +
                                         std::string(std::strerror(errno)));
            }
            cq_ring_ = sq_ring_;
            sq_ring_bytes_ = cq_ring_bytes_ = bytes;
            single_mapping_ = true;
        } else {
            sq_ring_ = ::mmap(nullptr, sq_ring_bytes_, PROT_READ | PROT_WRITE,
                              MAP_SHARED, ring_fd_, IORING_OFF_SQ_RING);
            cq_ring_ = ::mmap(nullptr, cq_ring_bytes_, PROT_READ | PROT_WRITE,
                              MAP_SHARED, ring_fd_, IORING_OFF_CQ_RING);
            if (sq_ring_ == MAP_FAILED || cq_ring_ == MAP_FAILED) {
                if (sq_ring_ == MAP_FAILED) sq_ring_ = nullptr;
                if (cq_ring_ == MAP_FAILED) cq_ring_ = nullptr;
                throw std::runtime_error("mmap io_uring queues failed: " +
                                         std::string(std::strerror(errno)));
            }
        }
        sqes_bytes_ = parameters.sq_entries * sizeof(io_uring_sqe);
        sqes_ = static_cast<io_uring_sqe*>(
            ::mmap(nullptr, sqes_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED,
                   ring_fd_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) {
            sqes_ = nullptr;
            throw std::runtime_error("mmap io_uring SQEs failed: " +
                                     std::string(std::strerror(errno)));
        }
        auto* sq = static_cast<std::byte*>(sq_ring_);
        auto* cq = static_cast<std::byte*>(cq_ring_);
        sq_head_ = reinterpret_cast<std::uint32_t*>(sq + parameters.sq_off.head);
        sq_tail_ = reinterpret_cast<std::uint32_t*>(sq + parameters.sq_off.tail);
        sq_ring_mask_ = reinterpret_cast<std::uint32_t*>(
            sq + parameters.sq_off.ring_mask);
        sq_entries_ = reinterpret_cast<std::uint32_t*>(
            sq + parameters.sq_off.ring_entries);
        sq_array_ = reinterpret_cast<std::uint32_t*>(sq + parameters.sq_off.array);
        cq_head_ = reinterpret_cast<std::uint32_t*>(cq + parameters.cq_off.head);
        cq_tail_ = reinterpret_cast<std::uint32_t*>(cq + parameters.cq_off.tail);
        cq_ring_mask_ = reinterpret_cast<std::uint32_t*>(
            cq + parameters.cq_off.ring_mask);
        cqes_ = reinterpret_cast<io_uring_cqe*>(cq + parameters.cq_off.cqes);
    }

    void allocate_buffers() {
        buffers_.resize(queue_depth_, nullptr);
        std::vector<iovec> vectors(queue_depth_);
        for (std::uint32_t index = 0; index < queue_depth_; ++index) {
            void* value = nullptr;
            const auto result = ::posix_memalign(&value, page_bytes_, page_bytes_);
            if (result != 0)
                throw std::runtime_error("allocate PLE registered buffer failed: " +
                                         std::string(std::strerror(result)));
            buffers_[index] = value;
            vectors[index].iov_base = value;
            vectors[index].iov_len = page_bytes_;
        }
        const auto result = static_cast<int>(::syscall(
            __NR_io_uring_register, ring_fd_, IORING_REGISTER_BUFFERS,
            vectors.data(), vectors.size()));
        if (result < 0)
            throw std::runtime_error("register PLE io_uring buffers failed: " +
                                     std::string(std::strerror(errno)));
        buffers_registered_ = true;
    }

    void reset() noexcept {
        if (buffers_registered_ && ring_fd_ >= 0)
            (void)::syscall(__NR_io_uring_register, ring_fd_,
                            IORING_UNREGISTER_BUFFERS, nullptr, 0);
        buffers_registered_ = false;
        for (auto* value : buffers_) std::free(value);
        buffers_.clear();
        if (sqes_) ::munmap(sqes_, sqes_bytes_);
        sqes_ = nullptr;
        if (single_mapping_) {
            if (sq_ring_) ::munmap(sq_ring_, sq_ring_bytes_);
        } else {
            if (sq_ring_) ::munmap(sq_ring_, sq_ring_bytes_);
            if (cq_ring_) ::munmap(cq_ring_, cq_ring_bytes_);
        }
        sq_ring_ = cq_ring_ = nullptr;
        if (ring_fd_ >= 0) ::close(ring_fd_);
        ring_fd_ = -1;
    }

    int ring_fd_ = -1;
    std::uint32_t queue_depth_ = 0;
    std::uint32_t page_bytes_ = 0;
    void* sq_ring_ = nullptr;
    void* cq_ring_ = nullptr;
    std::size_t sq_ring_bytes_ = 0;
    std::size_t cq_ring_bytes_ = 0;
    io_uring_sqe* sqes_ = nullptr;
    std::size_t sqes_bytes_ = 0;
    std::uint32_t* sq_head_ = nullptr;
    std::uint32_t* sq_tail_ = nullptr;
    std::uint32_t* sq_ring_mask_ = nullptr;
    std::uint32_t* sq_entries_ = nullptr;
    std::uint32_t* sq_array_ = nullptr;
    std::uint32_t* cq_head_ = nullptr;
    std::uint32_t* cq_tail_ = nullptr;
    std::uint32_t* cq_ring_mask_ = nullptr;
    io_uring_cqe* cqes_ = nullptr;
    std::vector<void*> buffers_;
    bool single_mapping_ = false;
    bool buffers_registered_ = false;
#else
    IoUringProvider(std::uint32_t, std::uint32_t) {
        throw std::runtime_error("io_uring is only available on Linux");
    }
    std::uint32_t queue_depth() const { return 0; }
    std::uint64_t read(const std::vector<ReadRequest>&) {
        throw std::runtime_error("io_uring is only available on Linux");
    }
#endif
};

PleStore::PleStore(PleLayoutV1 layout, std::uint64_t cache_bytes,
                   std::uint32_t cache_page_bytes)
    : PleStore(std::move(layout),
               PleStoreOptionsV1{cache_bytes, cache_page_bytes,
                                 PleIoModeV1::kAuto, 64}) {}

PleStore::PleStore(PleLayoutV1 layout, PleStoreOptionsV1 options)
    : layout_(std::move(layout)) {
    if (options.io_mode != PleIoModeV1::kAuto &&
        options.io_mode != PleIoModeV1::kBuffered &&
        options.io_mode != PleIoModeV1::kIoUringDirect)
        throw std::invalid_argument("PLE I/O mode is invalid");
    if (options.queue_depth == 0 || options.queue_depth > 4096)
        throw std::invalid_argument("PLE I/O queue depth is invalid");
    if (options.cache_bytes) {
        if (options.cache_page_bytes == 0 ||
            (options.cache_page_bytes & (options.cache_page_bytes - 1u)) != 0 ||
            options.cache_bytes < options.cache_page_bytes)
            throw std::invalid_argument("PLE cache page/capacity is invalid");
        cache_ = std::make_unique<Cache>(options.cache_bytes,
                                         options.cache_page_bytes);
        cache_->stats.capacity_bytes = options.cache_bytes;
    }
    try {
        for (const auto& file : layout_.files) {
            const int fd = open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0)
                throw std::runtime_error("cannot open PLE file " + file.path +
                                         ": " + std::strerror(errno));
            files_.push_back(OpenFile{file.index, fd, -1});
        }
        if (layout_.storage_dtype == DType::kFp8E4M3) {
            scale_parts_.resize(layout_.parts.size());
            for (const auto& part : layout_.parts) {
                auto& values = scale_parts_[part.logical_part].values;
                values.resize(static_cast<std::size_t>(part.rows));
                scale_resident_bytes_ +=
                    static_cast<std::uint64_t>(values.size()) *
                    sizeof(std::uint16_t);
                std::size_t done = 0;
                const auto bytes = static_cast<std::size_t>(part.scale_bytes);
                auto* output = reinterpret_cast<std::uint8_t*>(values.data());
                while (done < bytes) {
                    const auto count = pread(
                        file_descriptor(part.scale_file_index), output + done,
                        bytes - done,
                        static_cast<off_t>(part.scale_file_offset + done));
                    if (count < 0 && errno == EINTR) continue;
                    if (count <= 0)
                        throw std::runtime_error(
                            "PLE row-scale read failed: " +
                            std::string(std::strerror(errno)));
                    done += static_cast<std::size_t>(count);
                }
            }
        }

        const bool direct_candidate =
            options.io_mode != PleIoModeV1::kBuffered && cache_ &&
            cache_->page_bytes == 4096;
        if (options.io_mode == PleIoModeV1::kIoUringDirect &&
            !direct_candidate)
            throw std::invalid_argument(
                "direct PLE I/O requires a nonzero 4 KiB page cache");
        if (direct_candidate) {
            try {
                io_uring_ = std::make_unique<IoUringProvider>(
                    options.queue_depth, cache_->page_bytes);
#ifdef O_DIRECT
                for (auto& file : files_) {
                    const auto& metadata = layout_.files.at(file.index);
                    file.direct_fd =
                        open(metadata.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                    if (file.direct_fd < 0)
                        throw std::runtime_error(
                            "open PLE O_DIRECT file failed: " +
                            std::string(std::strerror(errno)));
                }
#else
                throw std::runtime_error("O_DIRECT is unavailable on this host");
#endif
                cache_->stats.io_uring_enabled = 1;
                cache_->stats.direct_io_enabled = 1;
            } catch (...) {
                for (auto& file : files_) {
                    if (file.direct_fd >= 0) close(file.direct_fd);
                    file.direct_fd = -1;
                }
                io_uring_.reset();
                if (options.io_mode == PleIoModeV1::kIoUringDirect) throw;
            }
        }
    } catch (...) {
        for (const auto& file : files_) {
            if (file.direct_fd >= 0) close(file.direct_fd);
            if (file.fd >= 0) close(file.fd);
        }
        throw;
    }
}

PleStore::~PleStore() {
    io_uring_.reset();
    for (const auto& file : files_) {
        if (file.direct_fd >= 0) close(file.direct_fd);
        if (file.fd >= 0) close(file.fd);
    }
}

const PlePartV1& PleStore::find_part(std::uint64_t global_row) const {
    if (global_row >= layout_.usable_rows)
        throw std::runtime_error("PLE row is outside usable vocabulary");
    const auto found = std::upper_bound(
        layout_.parts.begin(), layout_.parts.end(), global_row,
        [](std::uint64_t row, const PlePartV1& part) {
            return row < part.global_row_start;
        });
    if (found == layout_.parts.begin())
        throw std::runtime_error("PLE row has no logical part");
    const auto& part = *std::prev(found);
    if (global_row >= part.global_row_start + part.rows)
        throw std::runtime_error("PLE row falls in a logical part gap");
    return part;
}

int PleStore::file_descriptor(std::uint32_t file_index) const {
    const auto found = std::find_if(files_.begin(), files_.end(),
                                    [file_index](const OpenFile& file) {
                                        return file.index == file_index;
                                    });
    if (found == files_.end()) throw std::runtime_error("PLE file is not open");
    return found->fd;
}

std::uint64_t PleStore::file_size(std::uint32_t file_index) const {
    const auto found = std::find_if(layout_.files.begin(), layout_.files.end(),
                                    [file_index](const PleFileV1& file) {
                                        return file.index == file_index;
                                    });
    if (found == layout_.files.end())
        throw std::runtime_error("PLE physical file metadata is missing");
    return found->file_bytes;
}

void PleStore::read_exact(std::uint32_t file_index, std::uint64_t offset,
                          std::uint8_t* output, std::size_t size) const {
    read_extents({PleReadExtentV1{file_index, offset, 0, size}}, output, size);
}

void PleStore::read_extents(const std::vector<PleReadExtentV1>& extents,
                            std::uint8_t* output,
                            std::size_t output_bytes) const {
    for (const auto& extent : extents) {
        if (extent.bytes == 0 || extent.output_offset > output_bytes ||
            extent.bytes > output_bytes - extent.output_offset ||
            extent.offset > file_size(extent.file_index) ||
            extent.bytes > file_size(extent.file_index) - extent.offset)
            throw std::invalid_argument("PLE read extent is invalid");
    }
    if ((!extents.empty() && !output) || (extents.empty() && output_bytes != 0))
        throw std::invalid_argument("PLE read output is invalid");
    if (!cache_) {
        for (const auto& extent : extents) {
            std::size_t done = 0;
            while (done < extent.bytes) {
                const auto count = pread(
                    file_descriptor(extent.file_index),
                    output + extent.output_offset + done, extent.bytes - done,
                    static_cast<off_t>(extent.offset + done));
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0)
                    throw std::runtime_error("PLE read failed: " +
                                             std::string(std::strerror(errno)));
                done += static_cast<std::size_t>(count);
            }
        }
        return;
    }

    struct PageLoad {
        std::uint64_t key = 0;
        std::uint32_t file_index = 0;
        std::uint64_t page_offset = 0;
        std::size_t physical_bytes = 0;
        std::vector<std::uint8_t> bytes;
    };
    struct PendingCopy {
        std::size_t load = 0;
        std::size_t in_page = 0;
        std::size_t output_offset = 0;
        std::size_t bytes = 0;
    };

    std::lock_guard<std::mutex> lock(cache_->mutex);
    std::vector<PageLoad> loads;
    std::vector<PendingCopy> pending;
    std::unordered_map<std::uint64_t, std::size_t> load_by_key;
    std::unordered_set<std::uint64_t> request_pages;
    for (const auto& extent : extents) {
        std::size_t copied = 0;
        while (copied < extent.bytes) {
            const auto absolute = extent.offset + copied;
            const auto page_offset = absolute &
                ~static_cast<std::uint64_t>(cache_->page_bytes - 1u);
            const auto in_page = static_cast<std::size_t>(absolute - page_offset);
            const auto take = std::min<std::size_t>(
                extent.bytes - copied, cache_->page_bytes - in_page);
            if (extent.file_index >= 256)
                throw std::runtime_error("PLE cache file index exceeds V1 key space");
            const auto key = (static_cast<std::uint64_t>(extent.file_index) << 56u) |
                             (page_offset / cache_->page_bytes);
            request_pages.insert(key);
            auto found = cache_->entries.find(key);
            if (found != cache_->entries.end()) {
                ++cache_->stats.hits;
                cache_->lru.erase(found->second.lru);
                cache_->lru.push_front(key);
                found->second.lru = cache_->lru.begin();
                std::copy_n(found->second.bytes.data() + in_page, take,
                            output + extent.output_offset + copied);
            } else {
                auto queued = load_by_key.find(key);
                if (queued == load_by_key.end()) {
                    ++cache_->stats.misses;
                    const auto available = file_size(extent.file_index) - page_offset;
                    PageLoad load;
                    load.key = key;
                    load.file_index = extent.file_index;
                    load.page_offset = page_offset;
                    load.physical_bytes = static_cast<std::size_t>(
                        std::min<std::uint64_t>(available, cache_->page_bytes));
                    load.bytes.resize(cache_->page_bytes, 0);
                    const auto index = loads.size();
                    loads.push_back(std::move(load));
                    queued = load_by_key.emplace(key, index).first;
                } else {
                    ++cache_->stats.hits;
                }
                pending.push_back(PendingCopy{
                    queued->second, in_page, extent.output_offset + copied, take});
            }
            copied += take;
        }
    }
    cache_->stats.unique_page_requests += request_pages.size();

    if (!loads.empty()) {
        const auto started = std::chrono::steady_clock::now();
        try {
            const auto maximum_batch = io_uring_
                ? static_cast<std::size_t>(io_uring_->queue_depth())
                : std::size_t{1};
            for (std::size_t first = 0; first < loads.size();
                 first += maximum_batch) {
                const auto count = std::min(maximum_batch, loads.size() - first);
                ++cache_->stats.read_batches;
                cache_->stats.maximum_queue_depth = std::max<std::uint64_t>(
                    cache_->stats.maximum_queue_depth, count);
                if (io_uring_) {
                    std::vector<IoUringProvider::ReadRequest> requests;
                    requests.reserve(count);
                    for (std::size_t local = 0; local < count; ++local) {
                        auto& load = loads[first + local];
                        const auto opened = std::find_if(
                            files_.begin(), files_.end(),
                            [&load](const OpenFile& file) {
                                return file.index == load.file_index;
                            });
                        if (opened == files_.end() || opened->direct_fd < 0)
                            throw std::runtime_error(
                                "PLE direct file descriptor is unavailable");
                        requests.push_back(IoUringProvider::ReadRequest{
                            opened->direct_fd, load.page_offset,
                            load.bytes.data(), load.physical_bytes});
                    }
                    const auto physical = io_uring_->read(requests);
                    cache_->stats.physical_read_bytes += physical;
                    cache_->stats.direct_read_bytes += physical;
                    cache_->stats.io_uring_submissions += count;
                    cache_->stats.io_uring_completions += count;
                } else {
                    auto& load = loads[first];
                    std::size_t done = 0;
                    while (done < load.physical_bytes) {
                        const auto count_read = pread(
                            file_descriptor(load.file_index),
                            load.bytes.data() + done,
                            load.physical_bytes - done,
                            static_cast<off_t>(load.page_offset + done));
                        if (count_read < 0 && errno == EINTR) continue;
                        if (count_read <= 0)
                            throw std::runtime_error(
                                "PLE page read failed: " +
                                std::string(std::strerror(errno)));
                        done += static_cast<std::size_t>(count_read);
                    }
                    cache_->stats.physical_read_bytes += load.physical_bytes;
                }
            }
            cache_->stats.read_operations += loads.size();
            const auto latency = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count());
            if (cache_->read_latencies.size() == 4096)
                cache_->read_latencies.pop_front();
            cache_->read_latencies.push_back(latency);
        } catch (...) {
            ++cache_->stats.read_errors;
            throw;
        }

        for (const auto& copy : pending) {
            const auto& load = loads.at(copy.load);
            std::copy_n(load.bytes.data() + copy.in_page, copy.bytes,
                        output + copy.output_offset);
        }
        for (auto& load : loads) {
            if (cache_->entries.size() == cache_->max_pages) {
                const auto victim = cache_->lru.back();
                cache_->lru.pop_back();
                cache_->entries.erase(victim);
                ++cache_->stats.evictions;
            }
            cache_->lru.push_front(load.key);
            cache_->entries.emplace(
                load.key,
                Cache::Entry{std::move(load.bytes), cache_->lru.begin()});
        }
    }
}

std::vector<std::uint8_t> PleStore::read_row(std::uint64_t global_row) const {
    const auto& part = find_part(global_row);
    const auto local_row = global_row - part.global_row_start;
    const auto offset = part.file_offset +
                        local_row * layout_.row_stride_bytes;
    std::vector<std::uint8_t> result(layout_.row_stride_bytes);
    read_exact(part.file_index, offset, result.data(), result.size());
    return result;
}

PleCacheStatsV1 PleStore::cache_stats() const {
    if (!cache_) {
        PleCacheStatsV1 stats;
        stats.scale_resident_bytes = scale_resident_bytes_;
        return stats;
    }
    std::lock_guard<std::mutex> lock(cache_->mutex);
    auto stats = cache_->stats;
    stats.resident_bytes = cache_->entries.size() * cache_->page_bytes;
    stats.scale_resident_bytes = scale_resident_bytes_;
    if (!cache_->read_latencies.empty()) {
        auto values = std::vector<std::uint64_t>(cache_->read_latencies.begin(),
                                                 cache_->read_latencies.end());
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](std::size_t percent) {
            const auto rank = (values.size() * percent + 99) / 100;
            return values.at(rank - 1);
        };
        stats.read_latency_p50_ns = percentile(50);
        stats.read_latency_p95_ns = percentile(95);
        stats.read_latency_p99_ns = percentile(99);
    }
    return stats;
}

std::vector<std::uint8_t> PleStore::read_rows(
    const std::vector<std::uint64_t>& global_rows) const {
    if (global_rows.size() >
        std::numeric_limits<std::size_t>::max() / layout_.row_stride_bytes)
        throw std::overflow_error("PLE row batch byte size overflows");
    std::vector<std::uint8_t> result(
        global_rows.size() * layout_.row_stride_bytes);
    read_rows_into(global_rows, result.data(), result.size());
    return result;
}

void PleStore::read_rows_into(
    const std::vector<std::uint64_t>& global_rows, std::uint8_t* output,
    std::size_t output_bytes) const {
    if (global_rows.size() >
        std::numeric_limits<std::size_t>::max() / layout_.row_stride_bytes)
        throw std::overflow_error("PLE row batch byte size overflows");
    const auto expected = global_rows.size() * layout_.row_stride_bytes;
    if ((expected != 0 && !output) || output_bytes != expected)
        throw std::invalid_argument("PLE output extent differs from row batch");
    std::vector<PleReadExtentV1> extents;
    extents.reserve(global_rows.size());
    for (std::size_t index = 0; index < global_rows.size(); ++index) {
        const auto& part = find_part(global_rows[index]);
        const auto local_row = global_rows[index] - part.global_row_start;
        const auto offset =
            part.file_offset + local_row * layout_.row_stride_bytes;
        extents.push_back(PleReadExtentV1{
            part.file_index, offset, index * layout_.row_stride_bytes,
            layout_.row_stride_bytes});
    }
    if (cache_) {
        std::lock_guard<std::mutex> lock(cache_->mutex);
        cache_->stats.requested_rows += global_rows.size();
        cache_->stats.useful_bytes += expected;
    }
    read_extents(extents, output, output_bytes);
}

void PleStore::read_row_scales_into(
    const std::vector<std::uint64_t>& global_rows, std::uint16_t* output,
    std::size_t output_scales) const {
    if ((global_rows.size() != 0 && !output) ||
        output_scales != global_rows.size())
        throw std::invalid_argument("PLE scale output extent differs");
    if (layout_.storage_dtype != DType::kFp8E4M3 ||
        scale_parts_.size() != layout_.parts.size())
        throw std::logic_error("PLE row scales are unavailable");
    for (std::size_t index = 0; index < global_rows.size(); ++index) {
        const auto& part = find_part(global_rows[index]);
        const auto local = global_rows[index] - part.global_row_start;
        output[index] =
            scale_parts_[part.logical_part].values[static_cast<std::size_t>(local)];
    }
}

}  // namespace q38
