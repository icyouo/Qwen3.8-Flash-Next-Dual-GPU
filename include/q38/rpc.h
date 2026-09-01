#ifndef Q38_RPC_H
#define Q38_RPC_H

#include "q38/contracts.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q38 {

class DualStageExecutor;

constexpr std::uint32_t kExecutorRpcMagic = fourcc('Q', '3', '8', 'R');
constexpr std::uint16_t kExecutorRpcVersion = 1;
constexpr std::uint64_t kExecutorRpcMaximumPayload = 64ull << 20u;
constexpr std::uint32_t kExecutorRpcFlagDeadline = 1u << 0u;

enum class ExecutorRpcOpcodeV1 : std::uint32_t {
    kInvalid = 0,
    kPing = 1,
    kState = 2,
    kAppend = 3,
    kSeed = 4,
    kDecode = 5,
    kSpeculative = 6,
    kStats = 7,
    // request_id identifies the active writer request to cancel.
    kCancel = 8,
    // Atomically drops mutable session state without unloading weights.
    kReset = 9,
};

enum class ExecutorRpcStatusV1 : std::uint32_t {
    kOk = 0,
    kBadRequest = 1,
    kFailedPrecondition = 2,
    kCapacityExhausted = 3,
    kInternal = 4,
    kPoisoned = 5,
    kCancelled = 6,
    kDeadlineExceeded = 7,
};

// All fields are native little-endian. The production host is x86_64 Linux;
// clients must use an explicit little-endian codec rather than mapping these
// structs directly. Append carries token_count int32 values. Decode uses
// argument0=steps. Speculative uses argument0=steps, argument1=max_draft.
// Writer requests may set kExecutorRpcFlagDeadline and timeout_ms. Cancel uses
// its request_id as the target writer request id.
struct alignas(8) ExecutorRpcRequestHeaderV1 {
    std::uint32_t magic = kExecutorRpcMagic;
    std::uint16_t version = kExecutorRpcVersion;
    std::uint16_t header_bytes = sizeof(ExecutorRpcRequestHeaderV1);
    std::uint64_t request_id = 0;
    std::uint64_t session_hash = 0;
    ExecutorRpcOpcodeV1 opcode = ExecutorRpcOpcodeV1::kInvalid;
    std::uint32_t flags = 0;
    std::uint32_t token_count = 0;
    std::uint32_t argument0 = 0;
    std::uint32_t argument1 = 0;
    std::uint32_t timeout_ms = 0;
    std::uint64_t payload_bytes = 0;
};

struct alignas(8) ExecutorRpcResponseHeaderV1 {
    std::uint32_t magic = kExecutorRpcMagic;
    std::uint16_t version = kExecutorRpcVersion;
    std::uint16_t header_bytes = sizeof(ExecutorRpcResponseHeaderV1);
    std::uint64_t request_id = 0;
    std::uint64_t session_hash = 0;
    ExecutorRpcStatusV1 status = ExecutorRpcStatusV1::kOk;
    ExecutorRpcOpcodeV1 opcode = ExecutorRpcOpcodeV1::kInvalid;
    std::uint32_t token_count = 0;
    std::uint32_t message_bytes = 0;
    std::uint64_t payload_bytes = 0;
    SessionFrontiersV1 frontiers{};
};

static_assert(sizeof(ExecutorRpcRequestHeaderV1) == 56,
              "executor request ABI changed without a version bump");
static_assert(sizeof(ExecutorRpcResponseHeaderV1) == 96,
              "executor response ABI changed without a version bump");

struct ExecutorRpcRequestV1 {
    ExecutorRpcRequestHeaderV1 header{};
    std::vector<std::int32_t> tokens;
};

struct ExecutorRpcResponseV1 {
    ExecutorRpcResponseHeaderV1 header{};
    std::vector<std::int32_t> tokens;
    std::string message;
};

bool validate_executor_rpc_request(const ExecutorRpcRequestV1& request,
                                   std::uint64_t expected_session_hash,
                                   std::string* error);

class ExecutorRpcServiceV1 {
public:
    ExecutorRpcServiceV1(DualStageExecutor* executor,
                         std::uint64_t session_hash,
                         std::string snapshot_journal_path = {});
    ~ExecutorRpcServiceV1();

    ExecutorRpcServiceV1(const ExecutorRpcServiceV1&) = delete;
    ExecutorRpcServiceV1& operator=(const ExecutorRpcServiceV1&) = delete;
    ExecutorRpcResponseV1 handle(const ExecutorRpcRequestV1& request) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Blocks in an accept loop. A disconnected client does not terminate the
// executor; the control plane may reconnect to the same socket and session.
void serve_executor_rpc_unix(ExecutorRpcServiceV1* service,
                             const std::string& socket_path);

}  // namespace q38

#endif
