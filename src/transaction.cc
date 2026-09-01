#include "q38/transaction.h"

#include <limits>

namespace q38 {

SessionTxnEngine::SessionTxnEngine(std::uint64_t session_hash)
    : session_hash_(session_hash) {}

bool SessionTxnEngine::fail(std::string* error, const char* message) const {
    if (error) *error = message;
    return false;
}

bool SessionTxnEngine::add_checked(std::uint64_t* value, std::uint64_t add,
                                   std::string* error) const {
    if (*value > std::numeric_limits<std::uint64_t>::max() - add)
        return fail(error, "frontier overflow");
    *value += add;
    return true;
}

bool SessionTxnEngine::append_known(std::uint32_t count, std::string* error) {
    if (active_) return fail(error, "cannot append during an active transaction");
    if (count == 0) return fail(error, "cannot append zero known tokens");
    return add_checked(&frontiers_.canonical, count, error);
}

bool SessionTxnEngine::prepare_append_known(std::uint32_t count,
                                            std::string* error) {
    return prepare_append_known(count, count, error);
}

bool SessionTxnEngine::prepare_append_known(std::uint32_t appended_count,
                                            std::uint32_t evaluated_count,
                                            std::string* error) {
    if (active_) return fail(error, "transaction already active");
    if (appended_count == 0)
        return fail(error, "cannot append zero known tokens");
    if (!validate_frontiers(frontiers_, error)) return false;
    const auto pending = frontiers_.canonical - frontiers_.target;
    if (pending > 1)
        return fail(error, "append request has too many pending tokens");
    if (evaluated_count != appended_count + pending)
        return fail(error, "append evaluated range does not cover pending suffix");
    const auto base_canonical = frontiers_.canonical;
    if (!add_checked(&frontiers_.canonical, appended_count, error)) return false;
    if (!prepare(TxnKind::kAppendKnown, evaluated_count, error)) {
        frontiers_.canonical = base_canonical;
        return false;
    }
    txn_.base_canonical = base_canonical;
    return true;
}

bool SessionTxnEngine::seed_decode_pending(std::string* error) {
    if (active_) return fail(error, "cannot seed during an active transaction");
    if (frontiers_.canonical != frontiers_.target)
        return fail(error, "decode seed requires no pending tokens");
    return add_checked(&frontiers_.canonical, 1, error);
}

bool SessionTxnEngine::prepare(TxnKind kind, std::uint32_t evaluated_count,
                               std::string* error) {
    if (active_) return fail(error, "transaction already active");
    if (evaluated_count == 0) return fail(error, "cannot evaluate zero tokens");
    if (!validate_frontiers(frontiers_, error)) return false;

    const std::uint64_t pending = frontiers_.canonical - frontiers_.target;
    switch (kind) {
    case TxnKind::kAppendKnown:
        if (evaluated_count > pending)
            return fail(error, "append transaction exceeds known pending tokens");
        break;
    case TxnKind::kDecode:
        if (pending != 1 || evaluated_count != 1)
            return fail(error, "decode requires exactly one pending token");
        break;
    case TxnKind::kSpeculative:
        if (pending != 1)
            return fail(error, "speculative verify requires one pending token");
        break;
    case TxnKind::kInvalid:
        return fail(error, "invalid transaction kind");
    }
    if (frontiers_.epoch == std::numeric_limits<std::uint64_t>::max())
        return fail(error, "transaction epoch exhausted");

    txn_ = SessionTxnV1{};
    txn_.session_hash = session_hash_;
    txn_.epoch = frontiers_.epoch + 1;
    txn_.base_target = frontiers_.target;
    txn_.base_canonical = frontiers_.canonical;
    txn_.evaluated_count = evaluated_count;
    txn_.kind = kind;
    txn_.status = TxnStatus::kPrepared;
    active_ = true;
    stage0_ack_ = false;
    stage1_ack_ = false;
    return true;
}

bool SessionTxnEngine::acknowledge(Stage stage, std::uint64_t epoch,
                                   std::uint32_t evaluated_count,
                                   std::string* error) {
    if (!active_) return fail(error, "no active transaction to acknowledge");
    if (txn_.status != TxnStatus::kPrepared &&
        txn_.status != TxnStatus::kDecided)
        return fail(error, "transaction is not acknowledgeable");
    if (epoch != txn_.epoch) return fail(error, "stage acknowledged wrong epoch");
    if (evaluated_count != txn_.evaluated_count)
        return fail(error, "stage acknowledged wrong evaluated count");
    if (stage == Stage::kStage0) {
        if (stage0_ack_) return fail(error, "duplicate stage0 acknowledgement");
        stage0_ack_ = true;
    } else if (stage == Stage::kStage1) {
        if (stage1_ack_) return fail(error, "duplicate stage1 acknowledgement");
        stage1_ack_ = true;
    } else {
        return fail(error, "invalid stage acknowledgement");
    }
    return true;
}

bool SessionTxnEngine::decide(std::uint32_t state_commit_count,
                              std::string* error) {
    if (!active_) return fail(error, "no active transaction to decide");
    if (txn_.status != TxnStatus::kPrepared)
        return fail(error, "transaction was already decided");
    switch (txn_.kind) {
    case TxnKind::kAppendKnown:
        if (state_commit_count != txn_.evaluated_count)
            return fail(error, "append must commit its full evaluated range");
        txn_.publish_count = 0;
        break;
    case TxnKind::kDecode:
        if (state_commit_count != 1)
            return fail(error, "plain decode must commit its pending token");
        txn_.publish_count = 1;
        break;
    case TxnKind::kSpeculative:
        if (state_commit_count == 0 ||
            state_commit_count > txn_.evaluated_count)
            return fail(error, "speculative accepted prefix is invalid");
        txn_.publish_count = state_commit_count;
        break;
    case TxnKind::kInvalid:
        return fail(error, "invalid transaction kind");
    }
    txn_.state_commit_count = state_commit_count;
    txn_.status = TxnStatus::kDecided;
    return true;
}

bool SessionTxnEngine::commit(CommitEventV1* event, std::string* error) {
    if (!active_) return fail(error, "no active transaction to commit");
    if (txn_.status != TxnStatus::kDecided)
        return fail(error, "transaction has no decision");
    if (!stage0_ack_ || !stage1_ack_)
        return fail(error, "both stages must acknowledge before commit");

    SessionFrontiersV1 next = frontiers_;
    if (!add_checked(&next.target, txn_.state_commit_count, error)) return false;
    next.stage0 = next.target;
    next.stage1 = next.target;
    if (txn_.kind == TxnKind::kDecode ||
        txn_.kind == TxnKind::kSpeculative) {
        if (!add_checked(&next.canonical, txn_.publish_count, error)) return false;
        next.draft = next.target;
    } else if (next.draft > next.target) {
        next.draft = next.target;
    }
    next.epoch = txn_.epoch;
    if (!validate_frontiers(next, error)) return false;

    txn_.status = TxnStatus::kCommitted;
    frontiers_ = next;
    if (event) {
        event->epoch = txn_.epoch;
        event->canonical = next.canonical;
        event->target = next.target;
        event->newly_published = txn_.publish_count;
        event->kind = txn_.kind;
    }
    active_ = false;
    return true;
}

bool SessionTxnEngine::rollback(std::string* error) {
    if (!active_) return fail(error, "no active transaction to roll back");
    txn_.status = TxnStatus::kRolledBack;
    if (txn_.kind == TxnKind::kAppendKnown)
        frontiers_.canonical = txn_.base_canonical;
    frontiers_.epoch = txn_.epoch;
    active_ = false;
    stage0_ack_ = false;
    stage1_ack_ = false;
    return validate_frontiers(frontiers_, error);
}

}  // namespace q38
