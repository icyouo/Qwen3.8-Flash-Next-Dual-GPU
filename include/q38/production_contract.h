#ifndef Q38_PRODUCTION_CONTRACT_H
#define Q38_PRODUCTION_CONTRACT_H

#include <cstdint>

namespace q38 {

constexpr char kQ38OfficialSourceRepo[] = "Qwen/Qwen3.8-Flash-Next";
constexpr char kQ38OfficialSourceCommit[] =
    "de4b8e4d43b917e7706784d8bb445c9af86a3540";
constexpr char kQ38AmperePolicySha256[] =
    "b670f647c3b628f9e671f9ad46bf794093e0149c0d4f1ea64c18d22453a35723";
constexpr std::uint32_t kQ38ProductionCut = 25;

}  // namespace q38

#endif
