#ifndef Q38_CUDA_TE_GRAPH_H
#define Q38_CUDA_TE_GRAPH_H
#include <cstddef>
#include <functional>
#include <memory>

namespace q38 {
// Owns graph executable resources; tensor and stream storage remain caller-owned.
// Construction records only. Paired replay composes both ranks' GPU pieces and
// collective dependencies; ordinary replay retains host collective dispatch.
// QSA/dynamic metadata boundaries stay outside either path.
class CudaTeGraphTape {
public:
    CudaTeGraphTape(int device, void* stream, const std::function<void()>& enqueue,
                    bool bulk_prefill = false);
    ~CudaTeGraphTape();
    CudaTeGraphTape(const CudaTeGraphTape&) = delete;
    CudaTeGraphTape& operator=(const CudaTeGraphTape&) = delete;
    void launch(void* stream, const std::function<void()>& check_cancel = {}) const;
    std::size_t nodes() const;
    std::size_t graphs() const;
    std::size_t collectives() const;
    std::size_t held_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
