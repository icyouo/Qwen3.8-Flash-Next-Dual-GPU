#ifndef Q38_SNAPSHOT_H
#define Q38_SNAPSHOT_H

#include "q38/contracts.h"
#include "q38/identity.h"
#include "q38/sampling.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q38 {

constexpr std::uint32_t kStateSnapshotMagic = fourcc('Q', '3', '8', 'N');
constexpr std::uint32_t kSnapshotColdRebuild = 1u << 0u;
constexpr std::uint32_t kSnapshotHasLastSample = 1u << 1u;

struct alignas(8) StateSnapshotHeaderV1 {
    std::uint32_t magic = kStateSnapshotMagic;
    std::uint16_t version = kContractVersion;
    std::uint16_t header_bytes = sizeof(StateSnapshotHeaderV1);
    std::uint64_t identity_checksum = 0;
    std::uint64_t sequence = 0;
    std::uint64_t timestamp_ns = 0;
    SessionFrontiersV1 frontiers{};
    SamplerStateV1 sampler{};
    std::uint64_t canonical_token_count = 0;
    std::uint64_t token_start = 0;
    std::uint64_t token_count = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t canonical_checksum = 0;
    std::int32_t last_sample = -1;
    std::uint32_t flags = kSnapshotColdRebuild;
    std::uint64_t record_checksum = 0;
};

struct StateSnapshotV1 {
    StateSnapshotHeaderV1 header{};
    // In-memory snapshots always expose the full canonical token vector. The
    // append-only journal stores only the suffix described by token_start.
    std::vector<std::int32_t> canonical_tokens;
};

static_assert(sizeof(StateSnapshotHeaderV1) == 176,
              "snapshot ABI changed without a version bump");

std::uint64_t canonical_token_checksum(
    const std::vector<std::int32_t>& tokens);
bool validate_state_snapshot(const StateSnapshotV1& snapshot,
                             const SessionIdentityV1& identity,
                             std::string* error);

class SnapshotJournal {
public:
    SnapshotJournal(std::string path, SessionIdentityV1 identity);
    ~SnapshotJournal();

    SnapshotJournal(const SnapshotJournal&) = delete;
    SnapshotJournal& operator=(const SnapshotJournal&) = delete;

    void append(StateSnapshotV1 snapshot);
    StateSnapshotV1 latest() const;
    bool empty() const;
    const std::string& path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace q38

#endif
