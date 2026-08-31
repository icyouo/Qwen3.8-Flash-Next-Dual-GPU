#ifndef Q38_IDENTITY_H
#define Q38_IDENTITY_H

#include "q38/contracts.h"
#include "q38/sampling.h"

#include <array>
#include <cstdint>
#include <string>

namespace q38 {

constexpr std::uint32_t kSessionIdentityMagic = fourcc('Q', '3', '8', 'I');
constexpr std::uint32_t kSessionIdentityDevelopment = 1u << 0u;

struct Digest256V1 {
    std::array<std::uint8_t, 32> bytes{};

    bool operator==(const Digest256V1& other) const {
        return bytes == other.bytes;
    }
    bool operator!=(const Digest256V1& other) const {
        return !(*this == other);
    }
};

struct alignas(8) SessionIdentityV1 {
    std::uint32_t magic = kSessionIdentityMagic;
    std::uint16_t version = kContractVersion;
    std::uint16_t header_bytes = sizeof(SessionIdentityV1);
    std::uint64_t session_hash = 0;
    Digest256V1 model_checkpoint{};
    Digest256V1 tokenizer{};
    Digest256V1 chat_template{};
    Digest256V1 runtime{};
    Digest256V1 kernels{};
    Digest256V1 ple_layout{};
    Digest256V1 stage_plan{};
    Digest256V1 sampling_parser{};
    std::uint32_t context_limit = 0;
    std::uint32_t flags = 0;
    std::uint64_t identity_checksum = 0;
};

static_assert(sizeof(Digest256V1) == 32, "digest ABI changed");
static_assert(sizeof(SessionIdentityV1) == 288,
              "session identity ABI changed without a version bump");

Digest256V1 parse_digest256(const std::string& hex);
std::string format_digest256(const Digest256V1& digest);
std::uint64_t session_identity_checksum(const SessionIdentityV1& identity);
bool validate_session_identity(const SessionIdentityV1& identity,
                               std::string* error);
SessionIdentityV1 load_session_identity(const std::string& path);
SessionIdentityV1 make_development_identity(
    std::uint64_t session_hash, std::uint32_t context_limit,
    const SamplingConfigV1& sampling);

}  // namespace q38

#endif
