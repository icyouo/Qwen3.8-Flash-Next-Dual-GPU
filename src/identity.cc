#include "q38/identity.h"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace q38 {

namespace {

int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

std::uint64_t parse_u64(const std::string& name, const std::string& value) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 0);
    if (consumed != value.size())
        throw std::runtime_error("invalid session identity integer: " + name);
    return parsed;
}

void hash_bytes(std::uint64_t* value, const void* input, std::size_t bytes) {
    const auto* data = static_cast<const std::uint8_t*>(input);
    for (std::size_t index = 0; index < bytes; ++index) {
        *value ^= data[index];
        *value *= UINT64_C(1099511628211);
    }
}

}  // namespace

Digest256V1 parse_digest256(const std::string& hex) {
    if (hex.size() != 64)
        throw std::invalid_argument("SHA-256 digest must contain 64 hex digits");
    Digest256V1 result;
    for (std::size_t index = 0; index < result.bytes.size(); ++index) {
        const auto high = hex_digit(hex[index * 2]);
        const auto low = hex_digit(hex[index * 2 + 1]);
        if (high < 0 || low < 0)
            throw std::invalid_argument("SHA-256 digest is not hexadecimal");
        result.bytes[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return result;
}

std::string format_digest256(const Digest256V1& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : digest.bytes)
        output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::uint64_t session_identity_checksum(const SessionIdentityV1& identity) {
    auto copy = identity;
    copy.identity_checksum = 0;
    std::uint64_t checksum = UINT64_C(1469598103934665603);
    hash_bytes(&checksum, &copy, sizeof(copy));
    return checksum;
}

bool validate_session_identity(const SessionIdentityV1& identity,
                               std::string* error) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (identity.magic != kSessionIdentityMagic ||
        identity.version != kContractVersion ||
        identity.header_bytes != sizeof(SessionIdentityV1))
        return fail("invalid session identity header");
    if (identity.session_hash == 0 || identity.context_limit == 0)
        return fail("session identity has zero session/context");
    if ((identity.flags & ~kSessionIdentityDevelopment) != 0)
        return fail("session identity flags are invalid");
    if (identity.identity_checksum != session_identity_checksum(identity))
        return fail("session identity checksum differs");
    if ((identity.flags & kSessionIdentityDevelopment) == 0) {
        const Digest256V1 zero{};
        if (identity.model_checkpoint == zero || identity.tokenizer == zero ||
            identity.chat_template == zero || identity.runtime == zero ||
            identity.kernels == zero || identity.ple_layout == zero ||
            identity.stage_plan == zero || identity.sampling_parser == zero)
            return fail("production session identity contains a zero digest");
    }
    return true;
}

SessionIdentityV1 load_session_identity(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open session identity: " + path);
    std::string line;
    if (!std::getline(input, line) || line != "Q38_SESSION_IDENTITY_V1")
        throw std::runtime_error("invalid session identity file header");
    std::unordered_map<std::string, std::string> values;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto split = line.find('=');
        if (split == std::string::npos || split == 0 ||
            !values.emplace(line.substr(0, split), line.substr(split + 1)).second)
            throw std::runtime_error("invalid/duplicate session identity field");
    }
    const std::array<const char*, 12> required{
        "session_hash", "model_checkpoint_sha256", "tokenizer_sha256",
        "chat_template_sha256", "runtime_sha256", "kernels_sha256",
        "ple_layout_sha256", "stage_plan_sha256",
        "sampling_parser_sha256", "context_limit", "flags",
        "identity_checksum"};
    for (const auto* key : required)
        if (values.count(key) == 0)
            throw std::runtime_error(std::string("missing identity field: ") +
                                     key);
    if (values.size() != required.size())
        throw std::runtime_error("unexpected session identity field");

    SessionIdentityV1 result;
    result.session_hash = parse_u64("session_hash", values.at("session_hash"));
    result.model_checkpoint =
        parse_digest256(values.at("model_checkpoint_sha256"));
    result.tokenizer = parse_digest256(values.at("tokenizer_sha256"));
    result.chat_template =
        parse_digest256(values.at("chat_template_sha256"));
    result.runtime = parse_digest256(values.at("runtime_sha256"));
    result.kernels = parse_digest256(values.at("kernels_sha256"));
    result.ple_layout = parse_digest256(values.at("ple_layout_sha256"));
    result.stage_plan = parse_digest256(values.at("stage_plan_sha256"));
    result.sampling_parser =
        parse_digest256(values.at("sampling_parser_sha256"));
    const auto context = parse_u64("context_limit", values.at("context_limit"));
    const auto flags = parse_u64("flags", values.at("flags"));
    if (context > std::numeric_limits<std::uint32_t>::max() ||
        flags > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("session identity value overflows");
    result.context_limit = static_cast<std::uint32_t>(context);
    result.flags = static_cast<std::uint32_t>(flags);
    result.identity_checksum =
        parse_u64("identity_checksum", values.at("identity_checksum"));
    std::string error;
    if (!validate_session_identity(result, &error))
        throw std::runtime_error("invalid session identity: " + error);
    return result;
}

SessionIdentityV1 make_development_identity(
    std::uint64_t session_hash, std::uint32_t context_limit,
    const SamplingConfigV1& sampling) {
    SessionIdentityV1 result;
    result.session_hash = session_hash;
    result.context_limit = context_limit;
    result.flags = kSessionIdentityDevelopment;
    // Development identities deliberately carry zero artifact digests but do
    // bind the sampling contract into the checksum.
    std::uint64_t sampling_hash = UINT64_C(1469598103934665603);
    hash_bytes(&sampling_hash, &sampling, sizeof(sampling));
    std::memcpy(result.sampling_parser.bytes.data(), &sampling_hash,
                sizeof(sampling_hash));
    result.identity_checksum = session_identity_checksum(result);
    return result;
}

}  // namespace q38
