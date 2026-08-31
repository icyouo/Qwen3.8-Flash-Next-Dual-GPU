#include "q38/rpc.h"

#include "q38/executor.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace q38 {

namespace {

// One target transaction per writer request preserves retry and cancellation
// atomicity. The built-in control plane batches requests, not semantic decode
// steps.
constexpr std::uint32_t kMaximumStepsPerRequest = 1;
constexpr std::uint32_t kMaximumDraftWidth = 64;
constexpr std::uint32_t kMaximumTimeoutMs = 24u * 60u * 60u * 1000u;
constexpr std::size_t kMaximumCachedWriterResponses = 4096;

bool fail(std::string* error, const char* message) {
    if (error) *error = message;
    return false;
}

bool valid_opcode(ExecutorRpcOpcodeV1 opcode) {
    switch (opcode) {
    case ExecutorRpcOpcodeV1::kPing:
    case ExecutorRpcOpcodeV1::kState:
    case ExecutorRpcOpcodeV1::kAppend:
    case ExecutorRpcOpcodeV1::kSeed:
    case ExecutorRpcOpcodeV1::kDecode:
    case ExecutorRpcOpcodeV1::kSpeculative:
    case ExecutorRpcOpcodeV1::kStats:
    case ExecutorRpcOpcodeV1::kCancel:
        return true;
    case ExecutorRpcOpcodeV1::kInvalid:
        return false;
    }
    return false;
}

bool is_writer_opcode(ExecutorRpcOpcodeV1 opcode) {
    return opcode == ExecutorRpcOpcodeV1::kAppend ||
           opcode == ExecutorRpcOpcodeV1::kSeed ||
           opcode == ExecutorRpcOpcodeV1::kDecode ||
           opcode == ExecutorRpcOpcodeV1::kSpeculative;
}

struct RequestFingerprint {
    std::uint64_t first = UINT64_C(1469598103934665603);
    std::uint64_t second = UINT64_C(1099511628211);

    bool operator==(const RequestFingerprint& other) const {
        return first == other.first && second == other.second;
    }
};

void fingerprint_bytes(RequestFingerprint* fingerprint, const void* data,
                       std::size_t bytes) {
    const auto* input = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        fingerprint->first ^= input[index];
        fingerprint->first *= UINT64_C(1099511628211);
        fingerprint->second += input[index] + UINT64_C(0x9e3779b97f4a7c15);
        fingerprint->second ^= fingerprint->second >> 29u;
        fingerprint->second *= UINT64_C(0xbf58476d1ce4e5b9);
    }
}

RequestFingerprint request_fingerprint(const ExecutorRpcRequestV1& request) {
    RequestFingerprint result;
    const auto& header = request.header;
    fingerprint_bytes(&result, &header.session_hash,
                      sizeof(header.session_hash));
    fingerprint_bytes(&result, &header.opcode, sizeof(header.opcode));
    fingerprint_bytes(&result, &header.flags, sizeof(header.flags));
    fingerprint_bytes(&result, &header.token_count,
                      sizeof(header.token_count));
    fingerprint_bytes(&result, &header.argument0, sizeof(header.argument0));
    fingerprint_bytes(&result, &header.argument1, sizeof(header.argument1));
    fingerprint_bytes(&result, &header.timeout_ms, sizeof(header.timeout_ms));
    if (!request.tokens.empty())
        fingerprint_bytes(&result, request.tokens.data(),
                          request.tokens.size() * sizeof(std::int32_t));
    return result;
}

ExecutorRpcResponseV1 response_base(const ExecutorRpcRequestV1& request,
                                    std::uint64_t session_hash) {
    ExecutorRpcResponseV1 response;
    response.header.request_id = request.header.request_id;
    response.header.session_hash = session_hash;
    response.header.opcode = request.header.opcode;
    return response;
}

void finalize_response(ExecutorRpcResponseV1* response,
                       const SessionFrontiersV1& frontiers) {
    response->header.frontiers = frontiers;
    response->header.token_count =
        static_cast<std::uint32_t>(response->tokens.size());
    response->header.message_bytes =
        static_cast<std::uint32_t>(response->message.size());
    response->header.payload_bytes =
        response->tokens.size() * sizeof(std::int32_t) +
        response->message.size();
}

class FileDescriptor {
public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    ~FileDescriptor() {
        if (value_ >= 0) (void)::close(value_);
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    int get() const { return value_; }

private:
    int value_;
};

bool read_exact_or_eof(int fd, void* output, std::size_t bytes) {
    auto* cursor = static_cast<std::byte*>(output);
    std::size_t done = 0;
    while (done < bytes) {
        const auto count = ::read(fd, cursor + done, bytes - done);
        if (count == 0) {
            if (done == 0) return false;
            throw std::runtime_error("executor RPC frame ended early");
        }
        if (count < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(),
                                    "read executor RPC");
        }
        done += static_cast<std::size_t>(count);
    }
    return true;
}

void write_exact(int fd, const void* input, std::size_t bytes) {
    const auto* cursor = static_cast<const std::byte*>(input);
    std::size_t done = 0;
    while (done < bytes) {
        const auto count =
            ::send(fd, cursor + done, bytes - done, MSG_NOSIGNAL);
        if (count < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(),
                                    "write executor RPC");
        }
        if (count == 0)
            throw std::runtime_error("executor RPC write made no progress");
        done += static_cast<std::size_t>(count);
    }
}

void write_response(int fd, const ExecutorRpcResponseV1& response) {
    write_exact(fd, &response.header, sizeof(response.header));
    if (!response.tokens.empty())
        write_exact(fd, response.tokens.data(),
                    response.tokens.size() * sizeof(std::int32_t));
    if (!response.message.empty())
        write_exact(fd, response.message.data(), response.message.size());
}

void serve_rpc_client(ExecutorRpcServiceV1* service, int accepted) noexcept {
    FileDescriptor client(accepted);
    try {
        for (;;) {
            ExecutorRpcRequestV1 request;
            if (!read_exact_or_eof(client.get(), &request.header,
                                   sizeof(request.header)))
                break;
            if (request.header.payload_bytes > kExecutorRpcMaximumPayload ||
                request.header.payload_bytes % sizeof(std::int32_t) != 0)
                throw std::runtime_error(
                    "executor RPC wire payload is invalid");
            request.tokens.resize(static_cast<std::size_t>(
                request.header.payload_bytes / sizeof(std::int32_t)));
            if (!request.tokens.empty() &&
                !read_exact_or_eof(
                    client.get(), request.tokens.data(),
                    request.tokens.size() * sizeof(std::int32_t)))
                throw std::runtime_error("executor RPC payload ended early");
            const auto response = service->handle(request);
            write_response(client.get(), response);
        }
    } catch (const std::exception&) {
        // A malformed or disconnected client cannot poison the executor.
    }
}

void remove_stale_socket(const std::string& path) {
    struct stat info {};
    if (::lstat(path.c_str(), &info) != 0) {
        if (errno == ENOENT) return;
        throw std::system_error(errno, std::generic_category(),
                                "inspect executor RPC socket");
    }
    if (!S_ISSOCK(info.st_mode))
        throw std::runtime_error(
            "executor RPC path exists and is not a socket");
    if (::unlink(path.c_str()) != 0)
        throw std::system_error(errno, std::generic_category(),
                                "remove stale executor RPC socket");
}

}  // namespace

bool validate_executor_rpc_request(const ExecutorRpcRequestV1& request,
                                   std::uint64_t expected_session_hash,
                                   std::string* error) {
    const auto& header = request.header;
    if (header.magic != kExecutorRpcMagic)
        return fail(error, "invalid executor RPC magic");
    if (header.version != kExecutorRpcVersion ||
        header.header_bytes != sizeof(ExecutorRpcRequestHeaderV1))
        return fail(error, "unsupported executor RPC version");
    if (header.session_hash != expected_session_hash)
        return fail(error, "executor RPC session does not match");
    if (header.request_id == 0)
        return fail(error, "executor RPC request id must be nonzero");
    if (!valid_opcode(header.opcode))
        return fail(error, "invalid executor RPC opcode");
    if ((header.flags & ~kExecutorRpcFlagDeadline) != 0)
        return fail(error, "executor RPC flags are invalid");
    if (is_writer_opcode(header.opcode)) {
        if ((header.flags & kExecutorRpcFlagDeadline) != 0) {
            if (header.timeout_ms == 0 ||
                header.timeout_ms > kMaximumTimeoutMs)
                return fail(error, "executor RPC timeout is invalid");
        } else if (header.timeout_ms != 0) {
            return fail(error, "executor RPC timeout requires deadline flag");
        }
    } else if (header.flags != 0 || header.timeout_ms != 0) {
        return fail(error, "read/cancel RPC cannot carry a deadline");
    }
    if (header.payload_bytes > kExecutorRpcMaximumPayload)
        return fail(error, "executor RPC payload is too large");
    if (request.tokens.size() != header.token_count)
        return fail(error, "executor RPC token count does not match payload");
    if (header.payload_bytes !=
        request.tokens.size() * sizeof(std::int32_t))
        return fail(error, "executor RPC payload byte count is invalid");

    switch (header.opcode) {
    case ExecutorRpcOpcodeV1::kAppend:
        if (request.tokens.empty())
            return fail(error, "append RPC requires tokens");
        if (header.argument0 != 0 || header.argument1 != 0)
            return fail(error, "append RPC arguments must be zero");
        break;
    case ExecutorRpcOpcodeV1::kDecode:
        if (!request.tokens.empty() || header.argument0 == 0 ||
            header.argument0 > kMaximumStepsPerRequest ||
            header.argument1 != 0)
            return fail(error, "decode RPC arguments are invalid");
        break;
    case ExecutorRpcOpcodeV1::kSpeculative:
        if (!request.tokens.empty() || header.argument0 == 0 ||
            header.argument0 > kMaximumStepsPerRequest ||
            header.argument1 == 0 ||
            header.argument1 > kMaximumDraftWidth)
            return fail(error, "speculative RPC arguments are invalid");
        break;
    case ExecutorRpcOpcodeV1::kPing:
    case ExecutorRpcOpcodeV1::kState:
    case ExecutorRpcOpcodeV1::kSeed:
    case ExecutorRpcOpcodeV1::kStats:
    case ExecutorRpcOpcodeV1::kCancel:
        if (!request.tokens.empty() || header.argument0 != 0 ||
            header.argument1 != 0)
            return fail(error, "executor RPC operation takes no payload");
        break;
    case ExecutorRpcOpcodeV1::kInvalid:
        return fail(error, "invalid executor RPC opcode");
    }
    return true;
}

struct ExecutorRpcServiceV1::Impl {
    struct CachedWriterResponse {
        RequestFingerprint fingerprint{};
        ExecutorRpcResponseV1 response{};
    };

    Impl(DualStageExecutor* value_executor, std::uint64_t value_session_hash,
         std::string snapshot_path)
        : executor(value_executor), session_hash(value_session_hash) {
        if (!executor)
            throw std::invalid_argument("executor RPC has no executor");
        published_frontiers = executor->frontiers();
        published_stats = executor->stats();
        published_poisoned = executor->poisoned();
        if (!snapshot_path.empty()) {
            snapshot_journal = std::make_unique<SnapshotJournal>(
                std::move(snapshot_path), executor->identity());
            if (!snapshot_journal->empty())
                throw std::runtime_error(
                    "existing snapshot journal requires cold rebuild or a new session");
        }
    }

    DualStageExecutor* executor;
    std::uint64_t session_hash;
    // writer_mutex serializes semantic mutations. state_mutex protects the
    // cancellation registry and idempotence cache and is never held while a
    // model pass is executing.
    std::mutex writer_mutex;
    std::mutex state_mutex;
    std::uint64_t highest_writer_request_id = 0;
    std::deque<std::uint64_t> writer_response_order;
    std::unordered_map<std::uint64_t, CachedWriterResponse> writer_responses;
    std::uint64_t active_writer_request_id = 0;
    std::shared_ptr<CancellationToken> active_cancellation;
    SessionFrontiersV1 published_frontiers{};
    ExecutorStatsV1 published_stats{};
    bool published_poisoned = false;
    std::unique_ptr<SnapshotJournal> snapshot_journal;
};

ExecutorRpcServiceV1::ExecutorRpcServiceV1(DualStageExecutor* executor,
                                           std::uint64_t session_hash,
                                           std::string snapshot_journal_path)
    : impl_(std::make_unique<Impl>(executor, session_hash,
                                   std::move(snapshot_journal_path))) {}

ExecutorRpcServiceV1::~ExecutorRpcServiceV1() = default;

ExecutorRpcResponseV1 ExecutorRpcServiceV1::handle(
    const ExecutorRpcRequestV1& request) noexcept {
    auto& state = *impl_;
    auto response = response_base(request, state.session_hash);
    try {
        std::string error;
        if (!validate_executor_rpc_request(request, state.session_hash,
                                           &error)) {
            response.header.status = ExecutorRpcStatusV1::kBadRequest;
            response.message = std::move(error);
            std::lock_guard<std::mutex> state_lock(state.state_mutex);
            finalize_response(&response, state.published_frontiers);
            return response;
        }

        if (request.header.opcode == ExecutorRpcOpcodeV1::kCancel) {
            std::lock_guard<std::mutex> state_lock(state.state_mutex);
            if (state.active_writer_request_id == request.header.request_id &&
                state.active_cancellation) {
                if (state.active_cancellation->request_cancel()) {
                    response.message = "cancellation requested";
                } else if (state.active_cancellation->sealed()) {
                    response.header.status =
                        ExecutorRpcStatusV1::kFailedPrecondition;
                    response.message = "writer passed its commit point";
                } else {
                    response.message = "cancellation was already requested";
                }
            } else {
                response.header.status =
                    ExecutorRpcStatusV1::kFailedPrecondition;
                response.message =
                    state.writer_responses.count(request.header.request_id)
                        ? "writer request already completed"
                        : "writer request is not active";
            }
            finalize_response(&response, state.published_frontiers);
            return response;
        }

        if (is_writer_opcode(request.header.opcode)) {
            // Only writer execution is serialized. A Cancel request never
            // takes this mutex and can interrupt the active transaction.
            std::unique_lock<std::mutex> writer_lock(state.writer_mutex);
            const auto fingerprint = request_fingerprint(request);
            bool execute = true;
            {
                std::lock_guard<std::mutex> state_lock(state.state_mutex);
                const auto found =
                    state.writer_responses.find(request.header.request_id);
                if (found != state.writer_responses.end()) {
                    if (found->second.fingerprint == fingerprint)
                        return found->second.response;
                    response.header.status = ExecutorRpcStatusV1::kBadRequest;
                    response.message =
                        "request id was already used with different contents";
                    finalize_response(&response, state.published_frontiers);
                    return response;
                }
                if (request.header.request_id <=
                    state.highest_writer_request_id) {
                    response.header.status =
                        ExecutorRpcStatusV1::kFailedPrecondition;
                    response.message = "writer request id is stale";
                    finalize_response(&response, state.published_frontiers);
                    return response;
                }
                state.highest_writer_request_id = request.header.request_id;
                if (state.published_poisoned) {
                    response.header.status = ExecutorRpcStatusV1::kPoisoned;
                    response.message = "executor is fail-closed";
                    execute = false;
                }
            }

            auto cancellation = std::make_shared<CancellationToken>(
                request.header.timeout_ms);
            if (execute) {
                std::lock_guard<std::mutex> state_lock(state.state_mutex);
                state.active_writer_request_id = request.header.request_id;
                state.active_cancellation = cancellation;
            }
            try {
                if (execute) {
                    switch (request.header.opcode) {
                    case ExecutorRpcOpcodeV1::kAppend:
                        state.executor->append(request.tokens, cancellation);
                        break;
                    case ExecutorRpcOpcodeV1::kSeed:
                        response.tokens.push_back(
                            state.executor->seed_decode(cancellation));
                        break;
                    case ExecutorRpcOpcodeV1::kDecode: {
                        auto result = state.executor->decode_one(cancellation);
                        response.tokens = std::move(result.published_tokens);
                        break;
                    }
                    case ExecutorRpcOpcodeV1::kSpeculative: {
                        auto result = state.executor->speculative_step(
                            request.header.argument1, cancellation);
                        response.tokens = std::move(result.published_tokens);
                        break;
                    }
                    default:
                        throw std::logic_error(
                            "non-writer entered writer dispatch");
                    }
                }
            } catch (const ExecutionDeadlineExceeded& caught) {
                response.header.status =
                    ExecutorRpcStatusV1::kDeadlineExceeded;
                response.message = caught.what();
            } catch (const ExecutionCancelled& caught) {
                response.header.status = ExecutorRpcStatusV1::kCancelled;
                response.message = caught.what();
            } catch (const std::length_error& caught) {
                response.header.status =
                    ExecutorRpcStatusV1::kCapacityExhausted;
                response.message = caught.what();
            } catch (const std::invalid_argument& caught) {
                response.header.status = ExecutorRpcStatusV1::kBadRequest;
                response.message = caught.what();
            } catch (const std::logic_error& caught) {
                response.header.status =
                    ExecutorRpcStatusV1::kFailedPrecondition;
                response.message = caught.what();
            } catch (const std::exception& caught) {
                response.header.status = state.executor->poisoned()
                                             ? ExecutorRpcStatusV1::kPoisoned
                                             : ExecutorRpcStatusV1::kInternal;
                response.message = caught.what();
            } catch (...) {
                response.header.status = ExecutorRpcStatusV1::kInternal;
                response.message = "unknown executor failure";
            }

            if (response.header.status == ExecutorRpcStatusV1::kOk &&
                state.snapshot_journal) {
                try {
                    state.snapshot_journal->append(state.executor->snapshot());
                } catch (const std::exception& caught) {
                    state.executor->fail_closed();
                    response.header.status = ExecutorRpcStatusV1::kPoisoned;
                    response.tokens.clear();
                    response.message =
                        std::string("durable snapshot failed; session isolated: ") +
                        caught.what();
                }
            }

            {
                std::lock_guard<std::mutex> state_lock(state.state_mutex);
                state.published_frontiers = state.executor->frontiers();
                state.published_stats = state.executor->stats();
                state.published_poisoned = state.executor->poisoned();
                if (state.active_writer_request_id ==
                    request.header.request_id) {
                    state.active_writer_request_id = 0;
                    state.active_cancellation.reset();
                }
                finalize_response(&response, state.published_frontiers);
                state.writer_response_order.push_back(
                    request.header.request_id);
                state.writer_responses.emplace(
                    request.header.request_id,
                    Impl::CachedWriterResponse{fingerprint, response});
                if (state.writer_response_order.size() >
                    kMaximumCachedWriterResponses) {
                    const auto evicted = state.writer_response_order.front();
                    state.writer_response_order.pop_front();
                    state.writer_responses.erase(evicted);
                }
            }
            return response;
        }

        std::unique_lock<std::mutex> writer_lock(state.writer_mutex);
        if (state.executor->poisoned()) {
            response.header.status = ExecutorRpcStatusV1::kPoisoned;
            response.message = "executor is fail-closed";
        } else if (request.header.opcode == ExecutorRpcOpcodeV1::kStats) {
            response.message = metrics_json(state.executor->metrics());
        }
        finalize_response(&response, state.executor->frontiers());
        return response;
    } catch (const std::exception& caught) {
        response.header.status = ExecutorRpcStatusV1::kInternal;
        response.message = caught.what();
    } catch (...) {
        response.header.status = ExecutorRpcStatusV1::kInternal;
        response.message = "unknown executor RPC service failure";
    }
    std::lock_guard<std::mutex> state_lock(state.state_mutex);
    finalize_response(&response, state.published_frontiers);
    return response;
}

void serve_executor_rpc_unix(ExecutorRpcServiceV1* service,
                             const std::string& socket_path) {
    if (!service) throw std::invalid_argument("executor RPC service is null");
    if (socket_path.empty())
        throw std::invalid_argument("executor RPC socket path is empty");
    sockaddr_un address{};
    if (socket_path.size() >= sizeof(address.sun_path))
        throw std::invalid_argument("executor RPC socket path is too long");
    remove_stale_socket(socket_path);

    FileDescriptor listener(::socket(AF_UNIX, SOCK_STREAM, 0));
    if (listener.get() < 0)
        throw std::system_error(errno, std::generic_category(),
                                "create executor RPC socket");
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address),
               sizeof(address)) != 0)
        throw std::system_error(errno, std::generic_category(),
                                "bind executor RPC socket");
    if (::chmod(socket_path.c_str(), S_IRUSR | S_IWUSR) != 0)
        throw std::system_error(errno, std::generic_category(),
                                "protect executor RPC socket");
    if (::listen(listener.get(), 8) != 0)
        throw std::system_error(errno, std::generic_category(),
                                "listen on executor RPC socket");

    for (;;) {
        const auto accepted = ::accept(listener.get(), nullptr, nullptr);
        if (accepted < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(),
                                    "accept executor RPC client");
        }
        try {
            std::thread(serve_rpc_client, service, accepted).detach();
        } catch (...) {
            (void)::close(accepted);
            throw;
        }
    }
}

}  // namespace q38
