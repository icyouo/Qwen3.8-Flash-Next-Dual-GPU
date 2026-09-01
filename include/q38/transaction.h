#ifndef Q38_TRANSACTION_H
#define Q38_TRANSACTION_H

#include "q38/contracts.h"

#include <cstdint>
#include <string>

namespace q38 {

struct CommitEventV1 {
    std::uint64_t epoch = 0;
    std::uint64_t canonical = 0;
    std::uint64_t target = 0;
    std::uint32_t newly_published = 0;
    TxnKind kind = TxnKind::kInvalid;
};

class SessionTxnEngine {
public:
    explicit SessionTxnEngine(std::uint64_t session_hash);

    const SessionFrontiersV1& frontiers() const { return frontiers_; }
    bool has_active_transaction() const { return active_; }
    const SessionTxnV1& active_transaction() const { return txn_; }

    bool append_known(std::uint32_t count, std::string* error);
    // Atomically reserves a known suffix and prepares one request-wide
    // transaction. evaluated_count may include the one already-canonical
    // pending model token that must be evaluated before the new suffix.
    // Rollback restores the pre-request canonical frontier.
    bool prepare_append_known(std::uint32_t count, std::string* error);
    bool prepare_append_known(std::uint32_t appended_count,
                              std::uint32_t evaluated_count,
                              std::string* error);
    bool seed_decode_pending(std::string* error);
    bool prepare(TxnKind kind, std::uint32_t evaluated_count,
                 std::string* error);
    // Retained-draft verification may stop at the first mismatch. Shorten the
    // still-prepared speculative extent before either stage acknowledges it;
    // other transaction kinds and extent growth remain invalid.
    bool shorten_speculative(std::uint32_t evaluated_count,
                             std::string* error);
    bool acknowledge(Stage stage, std::uint64_t epoch,
                     std::uint32_t evaluated_count, std::string* error);
    bool decide(std::uint32_t state_commit_count, std::string* error);
    bool commit(CommitEventV1* event, std::string* error);
    bool rollback(std::string* error);
    void reset();

private:
    bool fail(std::string* error, const char* message) const;
    bool add_checked(std::uint64_t* value, std::uint64_t add,
                     std::string* error) const;

    std::uint64_t session_hash_;
    SessionFrontiersV1 frontiers_{};
    SessionTxnV1 txn_{};
    bool active_ = false;
    bool stage0_ack_ = false;
    bool stage1_ack_ = false;
};

}  // namespace q38

#endif
