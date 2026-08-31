#include "q38/contracts.h"

#include <limits>
#include <stdexcept>

namespace q38 {

namespace {

bool fail(std::string* error, const char* message) {
    if (error) *error = message;
    return false;
}

std::uint64_t mix64(std::uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31u;
    return value;
}

}  // namespace

std::size_t dtype_bytes(DType dtype) {
    switch (dtype) {
    case DType::kBFloat16:
    case DType::kFloat16:
        return 2;
    case DType::kFloat32:
        return 4;
    case DType::kInt8:
    case DType::kFp8E4M3:
        return 1;
    case DType::kInvalid:
        return 0;
    }
    return 0;
}

std::uint64_t boundary_payload_checksum(const std::uint16_t* words,
                                        std::size_t word_count) {
    if (word_count != 0 && !words)
        throw std::invalid_argument("boundary checksum input is null");
    std::uint64_t checksum = mix64(
        static_cast<std::uint64_t>(word_count) ^
        UINT64_C(0x713338424f554e44));
    for (std::size_t index = 0; index < word_count; ++index) {
        const auto tagged =
            (static_cast<std::uint64_t>(words[index]) << 48u) ^
            static_cast<std::uint64_t>(index) ^
            UINT64_C(0x9e3779b97f4a7c15);
        checksum ^= mix64(tagged);
    }
    return checksum;
}

bool validate_boundary(const StageBoundaryFrameV1& frame,
                       std::uint32_t max_tokens,
                       std::string* error) {
    if (frame.magic != kBoundaryMagic) return fail(error, "bad boundary magic");
    if (frame.version != kContractVersion)
        return fail(error, "unsupported boundary version");
    if (frame.header_bytes != sizeof(StageBoundaryFrameV1))
        return fail(error, "boundary header size mismatch");
    if (frame.producer_status != ProducerStatus::kReady)
        return fail(error, "boundary is not ready");
    if (frame.token_count == 0 || frame.token_count > max_tokens)
        return fail(error, "boundary token count is outside the lane limit");
    if (frame.hidden_width == 0) return fail(error, "zero boundary width");
    const std::size_t element_bytes = dtype_bytes(frame.hidden_dtype);
    if (element_bytes == 0) return fail(error, "unsupported boundary dtype");
    const std::uint64_t width = frame.hidden_width;
    const std::uint64_t count = frame.token_count;
    if (width > std::numeric_limits<std::uint64_t>::max() / count ||
        width * count > std::numeric_limits<std::uint64_t>::max() /
                            element_bytes)
        return fail(error, "boundary payload size overflows");
    const std::uint64_t expected = width * count * element_bytes;
    if (frame.payload_bytes != expected)
        return fail(error, "boundary payload byte count mismatch");
    return true;
}

bool validate_frontiers(const SessionFrontiersV1& frontiers,
                        std::string* error) {
    if (frontiers.stage0 != frontiers.target ||
        frontiers.stage1 != frontiers.target)
        return fail(error, "stage and target frontiers diverged");
    if (frontiers.target > frontiers.canonical)
        return fail(error, "target frontier exceeds canonical frontier");
    if (frontiers.draft > frontiers.canonical)
        return fail(error, "draft frontier exceeds canonical frontier");
    return true;
}

}  // namespace q38
