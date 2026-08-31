#ifndef Q38_CONTRACTS_H
#define Q38_CONTRACTS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace q38 {

constexpr std::uint16_t kContractVersion = 1;
constexpr std::uint32_t fourcc(char a, char b, char c, char d) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8u) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16u) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24u);
}

constexpr std::uint32_t kBoundaryMagic = fourcc('Q', '3', '8', 'B');
constexpr std::uint32_t kTxnMagic = fourcc('Q', '3', '8', 'T');

enum class DType : std::uint16_t {
    kInvalid = 0,
    kBFloat16 = 1,
    kFloat32 = 2,
    kFloat16 = 3,
    kInt8 = 4,
    kFp8E4M3 = 5,
};

enum class ProducerStatus : std::uint16_t {
    kEmpty = 0,
    kReady = 1,
    kFailed = 2,
};

enum class TxnKind : std::uint16_t {
    kInvalid = 0,
    kAppendKnown = 1,
    kDecode = 2,
    kSpeculative = 3,
};

enum class TxnStatus : std::uint16_t {
    kEmpty = 0,
    kPrepared = 1,
    kDecided = 2,
    kCommitted = 3,
    kRolledBack = 4,
};

enum class Stage : std::uint8_t {
    kStage0 = 0,
    kStage1 = 1,
};

struct alignas(8) StageBoundaryFrameV1 {
    std::uint32_t magic = kBoundaryMagic;
    std::uint16_t version = kContractVersion;
    std::uint16_t header_bytes = sizeof(StageBoundaryFrameV1);
    std::uint64_t session_hash = 0;
    std::uint64_t epoch = 0;
    std::uint64_t token_start = 0;
    std::uint32_t token_count = 0;
    DType hidden_dtype = DType::kInvalid;
    ProducerStatus producer_status = ProducerStatus::kEmpty;
    std::uint32_t hidden_width = 0;
    std::uint32_t ring_slot = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t payload_checksum = 0;
};

struct alignas(8) SessionFrontiersV1 {
    std::uint64_t canonical = 0;
    std::uint64_t target = 0;
    std::uint64_t stage0 = 0;
    std::uint64_t stage1 = 0;
    std::uint64_t draft = 0;
    std::uint64_t epoch = 0;
};

struct alignas(8) SessionTxnV1 {
    std::uint32_t magic = kTxnMagic;
    std::uint16_t version = kContractVersion;
    std::uint16_t header_bytes = sizeof(SessionTxnV1);
    std::uint64_t session_hash = 0;
    std::uint64_t epoch = 0;
    std::uint64_t base_target = 0;
    std::uint32_t evaluated_count = 0;
    TxnKind kind = TxnKind::kInvalid;
    TxnStatus status = TxnStatus::kEmpty;
    std::uint32_t state_commit_count = 0;
    std::uint32_t publish_count = 0;
    // Canonical frontier before this request reserved any input tokens.  It
    // lets an append transaction roll the entire request back, even when the
    // backend evaluated it as several bounded workspace chunks.
    std::uint64_t base_canonical = 0;
};

static_assert(std::is_standard_layout<StageBoundaryFrameV1>::value,
              "boundary contract must be standard-layout");
static_assert(std::is_standard_layout<SessionFrontiersV1>::value,
              "frontier contract must be standard-layout");
static_assert(std::is_standard_layout<SessionTxnV1>::value,
              "transaction contract must be standard-layout");
static_assert(sizeof(StageBoundaryFrameV1) == 64,
              "boundary ABI changed without a version bump");
static_assert(sizeof(SessionFrontiersV1) == 48,
              "frontier ABI changed without a version bump");
static_assert(sizeof(SessionTxnV1) == 56,
              "transaction ABI changed without a version bump");

std::size_t dtype_bytes(DType dtype);
std::uint64_t boundary_payload_checksum(const std::uint16_t* words,
                                        std::size_t word_count);
bool validate_boundary(const StageBoundaryFrameV1& frame,
                       std::uint32_t max_tokens,
                       std::string* error);
bool validate_frontiers(const SessionFrontiersV1& frontiers,
                        std::string* error);

}  // namespace q38

#endif
