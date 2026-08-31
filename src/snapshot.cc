#include "q38/snapshot.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace q38 {

namespace {

void hash_bytes(std::uint64_t* value, const void* input, std::size_t bytes) {
    const auto* data = static_cast<const std::uint8_t*>(input);
    for (std::size_t index = 0; index < bytes; ++index) {
        *value ^= data[index];
        *value *= UINT64_C(1099511628211);
    }
}

std::uint64_t record_checksum(const StateSnapshotHeaderV1& header,
                              const std::vector<std::int32_t>& suffix) {
    auto copy = header;
    copy.record_checksum = 0;
    std::uint64_t checksum = UINT64_C(1469598103934665603);
    hash_bytes(&checksum, &copy, sizeof(copy));
    if (!suffix.empty())
        hash_bytes(&checksum, suffix.data(),
                   suffix.size() * sizeof(std::int32_t));
    return checksum;
}

void write_all(int descriptor, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const std::byte*>(data);
    std::size_t written = 0;
    while (written < bytes) {
        const auto count =
            ::write(descriptor, cursor + written, bytes - written);
        if (count < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(),
                                    "write snapshot journal");
        }
        if (count == 0)
            throw std::runtime_error("snapshot journal write made no progress");
        written += static_cast<std::size_t>(count);
    }
}

enum class ReadResult { kComplete, kEof, kPartial };

ReadResult read_record_bytes(int descriptor, void* data, std::size_t bytes) {
    auto* cursor = static_cast<std::byte*>(data);
    std::size_t consumed = 0;
    while (consumed < bytes) {
        const auto count = ::read(descriptor, cursor + consumed, bytes - consumed);
        if (count == 0)
            return consumed == 0 ? ReadResult::kEof : ReadResult::kPartial;
        if (count < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(),
                                    "read snapshot journal");
        }
        consumed += static_cast<std::size_t>(count);
    }
    return ReadResult::kComplete;
}

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

}  // namespace

std::uint64_t canonical_token_checksum(
    const std::vector<std::int32_t>& tokens) {
    std::uint64_t checksum = UINT64_C(1469598103934665603);
    if (!tokens.empty())
        hash_bytes(&checksum, tokens.data(),
                   tokens.size() * sizeof(std::int32_t));
    const auto count = static_cast<std::uint64_t>(tokens.size());
    hash_bytes(&checksum, &count, sizeof(count));
    return checksum;
}

bool validate_state_snapshot(const StateSnapshotV1& snapshot,
                             const SessionIdentityV1& identity,
                             std::string* error) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    const auto& header = snapshot.header;
    if (header.magic != kStateSnapshotMagic ||
        header.version != kContractVersion ||
        header.header_bytes != sizeof(StateSnapshotHeaderV1))
        return fail("invalid snapshot header");
    if (header.identity_checksum != identity.identity_checksum)
        return fail("snapshot identity differs");
    if (header.sequence == 0 ||
        header.canonical_token_count != snapshot.canonical_tokens.size() ||
        header.canonical_token_count > identity.context_limit)
        return fail("snapshot token extent is invalid");
    if (header.frontiers.canonical != header.canonical_token_count ||
        header.frontiers.target > header.frontiers.canonical ||
        header.frontiers.canonical - header.frontiers.target > 1)
        return fail("snapshot frontiers are not cold-rebuildable");
    std::string frontier_error;
    if (!validate_frontiers(header.frontiers, &frontier_error))
        return fail("snapshot stage frontiers diverge");
    if ((header.flags & kSnapshotColdRebuild) == 0 ||
        (header.flags & ~(kSnapshotColdRebuild |
                          kSnapshotHasLastSample)) != 0)
        return fail("snapshot flags are invalid");
    if (header.canonical_checksum !=
        canonical_token_checksum(snapshot.canonical_tokens))
        return fail("snapshot canonical checksum differs");
    if (header.sampler.magic != kSamplerStateMagic ||
        header.sampler.canonical_tokens != header.canonical_token_count)
        return fail("snapshot sampler state differs from canonical tokens");
    if ((header.flags & kSnapshotHasLastSample) != 0 &&
        header.last_sample < 0)
        return fail("snapshot last sample flag is inconsistent");
    return true;
}

struct SnapshotJournal::Impl {
    Impl(std::string value_path, SessionIdentityV1 value_identity)
        : path(std::move(value_path)), identity(value_identity) {
        std::string error;
        if (path.empty() || !validate_session_identity(identity, &error))
            throw std::invalid_argument("invalid snapshot journal identity/path");
        load();
    }

    void load() {
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            if (errno == ENOENT) return;
            throw std::system_error(errno, std::generic_category(),
                                    "open snapshot journal");
        }
        try {
            std::vector<std::int32_t> canonical;
            std::uint64_t last_sequence = 0;
            std::uint64_t valid_bytes = 0;
            for (;;) {
                StateSnapshotHeaderV1 header;
                const auto header_result =
                    read_record_bytes(descriptor, &header, sizeof(header));
                if (header_result == ReadResult::kEof ||
                    header_result == ReadResult::kPartial)
                    break;
                if (header.magic != kStateSnapshotMagic ||
                    header.version != kContractVersion ||
                    header.header_bytes != sizeof(header) ||
                    header.identity_checksum != identity.identity_checksum ||
                    header.sequence <= last_sequence ||
                    header.token_start > canonical.size() ||
                    header.token_count > identity.context_limit ||
                    header.token_start + header.token_count !=
                        header.canonical_token_count ||
                    header.payload_bytes !=
                        header.token_count * sizeof(std::int32_t))
                    throw std::runtime_error("corrupt snapshot journal header");
                std::vector<std::int32_t> suffix(
                    static_cast<std::size_t>(header.token_count));
                const auto payload_result = read_record_bytes(
                    descriptor, suffix.data(),
                    suffix.size() * sizeof(std::int32_t));
                if (payload_result != ReadResult::kComplete) break;
                if (header.record_checksum !=
                    record_checksum(header, suffix))
                    throw std::runtime_error("snapshot record checksum differs");
                canonical.resize(static_cast<std::size_t>(header.token_start));
                canonical.insert(canonical.end(), suffix.begin(), suffix.end());
                StateSnapshotV1 candidate{header, canonical};
                std::string error;
                if (!validate_state_snapshot(candidate, identity, &error))
                    throw std::runtime_error("invalid snapshot journal state: " +
                                             error);
                current = std::move(candidate);
                last_sequence = header.sequence;
                valid_bytes += sizeof(header) + header.payload_bytes;
            }
            if (current.header.sequence != 0)
                current.header.token_start = 0;
            const auto end = ::lseek(descriptor, 0, SEEK_END);
            if (end < 0)
                throw std::system_error(errno, std::generic_category(),
                                        "measure snapshot journal");
            if (static_cast<std::uint64_t>(end) > valid_bytes &&
                ::truncate(path.c_str(), static_cast<off_t>(valid_bytes)) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "truncate partial snapshot tail");
        } catch (...) {
            (void)::close(descriptor);
            throw;
        }
        (void)::close(descriptor);
    }

    std::string path;
    SessionIdentityV1 identity{};
    StateSnapshotV1 current{};
    mutable std::mutex mutex;
};

SnapshotJournal::SnapshotJournal(std::string path, SessionIdentityV1 identity)
    : impl_(std::make_unique<Impl>(std::move(path), identity)) {}
SnapshotJournal::~SnapshotJournal() = default;

void SnapshotJournal::append(StateSnapshotV1 snapshot) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    snapshot.header.identity_checksum = impl_->identity.identity_checksum;
    snapshot.header.sequence = impl_->current.header.sequence + 1;
    snapshot.header.timestamp_ns = now_ns();
    snapshot.header.canonical_token_count = snapshot.canonical_tokens.size();
    snapshot.header.canonical_checksum =
        canonical_token_checksum(snapshot.canonical_tokens);
    std::size_t common = 0;
    while (common < impl_->current.canonical_tokens.size() &&
           common < snapshot.canonical_tokens.size() &&
           impl_->current.canonical_tokens[common] ==
               snapshot.canonical_tokens[common])
        ++common;
    snapshot.header.token_start = common;
    snapshot.header.token_count = snapshot.canonical_tokens.size() - common;
    snapshot.header.payload_bytes =
        snapshot.header.token_count * sizeof(std::int32_t);
    std::vector<std::int32_t> suffix(
        snapshot.canonical_tokens.begin() + static_cast<std::ptrdiff_t>(common),
        snapshot.canonical_tokens.end());
    snapshot.header.record_checksum =
        record_checksum(snapshot.header, suffix);
    std::string error;
    if (!validate_state_snapshot(snapshot, impl_->identity, &error))
        throw std::invalid_argument("cannot append invalid snapshot: " + error);

    const int descriptor = ::open(impl_->path.c_str(),
                                  O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                                  S_IRUSR | S_IWUSR);
    if (descriptor < 0)
        throw std::system_error(errno, std::generic_category(),
                                "open snapshot journal for append");
    try {
        write_all(descriptor, &snapshot.header, sizeof(snapshot.header));
        if (!suffix.empty())
            write_all(descriptor, suffix.data(),
                      suffix.size() * sizeof(std::int32_t));
        if (::fdatasync(descriptor) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "sync snapshot journal");
    } catch (...) {
        (void)::close(descriptor);
        throw;
    }
    (void)::close(descriptor);
    snapshot.header.token_start = 0;
    snapshot.header.token_count = snapshot.canonical_tokens.size();
    snapshot.header.payload_bytes =
        snapshot.canonical_tokens.size() * sizeof(std::int32_t);
    impl_->current = std::move(snapshot);
}

StateSnapshotV1 SnapshotJournal::latest() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->current.header.sequence == 0)
        throw std::runtime_error("snapshot journal is empty");
    return impl_->current;
}

bool SnapshotJournal::empty() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->current.header.sequence == 0;
}

const std::string& SnapshotJournal::path() const { return impl_->path; }

}  // namespace q38
