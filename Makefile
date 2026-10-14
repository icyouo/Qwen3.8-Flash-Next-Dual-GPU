CXX ?= c++
PYTHON ?= python3
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -Iinclude

CUDA_HOME ?= /usr/local/cuda-13.3
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOST_CXX ?= g++-14
CUDA_TARGET_INCLUDE ?= $(CUDA_HOME)/targets/x86_64-linux/include
# The production CMP 170HX boards report GA100 compute capability 8.0.
# Keep this overridable for development hosts with a different CUDA target.
CUDA_ARCH ?= sm_80
NVCCFLAGS ?= -O3 -lineinfo -std=c++17 -arch=$(CUDA_ARCH)
CUDA_EXTRA_FLAGS ?= --expt-relaxed-constexpr -Icuda/vendor/marlin
CUDA_COMPAT ?= 1
CUDA_COMPAT_FLAGS = $(if $(filter 1,$(CUDA_COMPAT)),-I$(BUILD_DIR)/cuda-compat,)
CUDA_LIBS ?= -lcublas

BUILD_DIR := build
RUNTIME_SOURCES := src/contracts.cc src/identity.cc src/snapshot.cc src/metrics.cc src/artifact.cc src/tensor_index.cc src/mapped_weights.cc src/device_artifact.cc src/model_plan.cc src/ple.cc src/ple_backend.cc src/sampling.cc src/transaction.cc src/mock_backend.cc src/executor.cc src/rpc.cc
HEADERS := $(wildcard include/q38/*.h) $(shell find cuda -name "*.cuh" -o -name "*.h" -o -name "*.hpp")
TEST_BIN := $(BUILD_DIR)/q38_runtime_tests
CLI_BIN := $(BUILD_DIR)/q38-runtime
CUDA_CHECK_BIN := $(BUILD_DIR)/q38_cuda_compile_check
CUDA_RUNTIME_BIN := $(BUILD_DIR)/q38-cuda-runtime
CUDA_TEST_BIN := $(BUILD_DIR)/q38_cuda_kernel_tests
CUDA_DECODE_BENCH_BIN := $(BUILD_DIR)/q38_decode_bench
CUDA_MOE_BENCH_BIN := $(BUILD_DIR)/q38_moe_bench
CUDA_COMPAT_STAMP := $(BUILD_DIR)/cuda-compat/.stamp
CUDA_RUNTIME_SOURCES := cuda/cuda_transport.cu cuda/cuda_te.cu cuda/cuda_te_graph.cu cuda/cuda_tp_backend.cc cuda/cuda_weights.cu cuda/q38_kernels.cu cuda/q38_sampling.cu cuda/q38_gdn.cu cuda/q38_qsa.cu cuda/q38_hyper.cu cuda/q38_moe.cu cuda/q38_moe_prefill.cu cuda/q38_marlin.cu cuda/q38_ple.cu cuda/cuda_backend.cu

.PHONY: all test python-test verify cuda-check cuda-runtime cuda-test cuda-bench strict-gate clean rank-worker-test

all: test

$(BUILD_DIR):
	mkdir -p $@

$(TEST_BIN): $(RUNTIME_SOURCES) tests/test_runtime.cc $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cc,$^) -o $@

$(CLI_BIN): $(RUNTIME_SOURCES) src/main.cc $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cc,$^) -o $@

$(CUDA_COMPAT_STAMP): tools/q38_cuda_compat.py | $(BUILD_DIR)
ifeq ($(CUDA_COMPAT),1)
	$(PYTHON) $< \
		--source-root $(CUDA_TARGET_INCLUDE) \
		--output-root $(BUILD_DIR)/cuda-compat
else
	mkdir -p $(BUILD_DIR)/cuda-compat
	touch $@
endif

$(CUDA_CHECK_BIN): $(CUDA_RUNTIME_SOURCES) cuda/cuda_compile_check.cc $(RUNTIME_SOURCES) $(CUDA_COMPAT_STAMP) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu %.cc,$^) -o $@ $(CUDA_LIBS)

$(CUDA_RUNTIME_BIN): $(CUDA_RUNTIME_SOURCES) cuda/cuda_main.cc $(RUNTIME_SOURCES) $(CUDA_COMPAT_STAMP) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu %.cc,$^) -o $@ $(CUDA_LIBS)

$(CUDA_TEST_BIN): cuda/cuda_transport.cu cuda/q38_hyper.cu cuda/q38_kernels.cu cuda/q38_sampling.cu cuda/q38_gdn.cu cuda/q38_qsa.cu cuda/q38_moe.cu cuda/q38_moe_prefill.cu cuda/q38_ple.cu cuda/cuda_kernel_tests.cu src/contracts.cc $(CUDA_COMPAT_STAMP) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu %.cc,$^) -o $@ $(CUDA_LIBS)

$(CUDA_DECODE_BENCH_BIN): cuda/q38_kernels.cu cuda/q38_decode_bench.cu $(CUDA_COMPAT_STAMP) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu %.cc,$^) -o $@

$(CUDA_MOE_BENCH_BIN): cuda/q38_moe.cu cuda/q38_moe_bench.cu $(CUDA_COMPAT_STAMP) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu %.cc,$^) -o $@

$(BUILD_DIR)/q38_rank_worker_tests: tests/test_paired_rank_worker.cc include/q38/paired_rank_worker.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread $< -o $@

rank-worker-test: $(BUILD_DIR)/q38_rank_worker_tests
	./$(BUILD_DIR)/q38_rank_worker_tests

test: $(TEST_BIN) $(CLI_BIN) rank-worker-test
	./$(TEST_BIN)
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

python-test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

verify: test cuda-check cuda-runtime

strict-gate:
	test -n "$(SOCKET)"
	test -n "$(SESSION_HASH)"
	test -n "$(EVIDENCE)"
	$(PYTHON) tools/q38_strict_gate.py --socket "$(SOCKET)" \
		--session-hash "$(SESSION_HASH)" --output "$(EVIDENCE)"

cuda-check: $(CUDA_CHECK_BIN)

cuda-runtime: $(CUDA_RUNTIME_BIN)

cuda-test: $(CUDA_TEST_BIN)
	./$(CUDA_TEST_BIN)

.PHONY: prefix-snapshot-test
.PHONY: te-graph-test
.PHONY: bulk-copy-test
.PHONY: p2p-bench collective-bench
p2p-bench: $(BUILD_DIR)/q38_p2p_bench
	./$(BUILD_DIR)/q38_p2p_bench

$(BUILD_DIR)/q38_p2p_bench: cuda/q38_p2p_bench.cu $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $< -o $@

# Optional NCCL reference: add -DQ38_P2P_BENCH_NCCL and its include/library paths.
P2P_BENCH_CPPFLAGS ?=
P2P_BENCH_LIBS ?=
collective-bench: $(BUILD_DIR)/q38_collective_bench
	./$(BUILD_DIR)/q38_collective_bench

$(BUILD_DIR)/q38_collective_bench: cuda/cuda_te.cu cuda/q38_collective_bench.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(P2P_BENCH_CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $(filter %.cu,$^) -o $@ -lcublas $(P2P_BENCH_LIBS)

bulk-copy-test: $(BUILD_DIR)/q38_bulk_copy_tests
	./$(BUILD_DIR)/q38_bulk_copy_tests

$(BUILD_DIR)/q38_bulk_copy_tests: cuda/cuda_te.cu cuda/q38_bulk_copy_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $(filter %.cu,$^) -o $@ -lcublas

te-graph-test: $(BUILD_DIR)/q38_te_graph_tests
	./$(BUILD_DIR)/q38_te_graph_tests

$(BUILD_DIR)/q38_te_graph_tests: cuda/cuda_te.cu cuda/cuda_te_graph.cu cuda/cuda_te_graph_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(filter %.cu,$^) -o $@ -lcublas

prefix-snapshot-test: $(BUILD_DIR)/q38_prefix_snapshot_tests
	./$(BUILD_DIR)/q38_prefix_snapshot_tests

$(BUILD_DIR)/q38_prefix_snapshot_tests: cuda/q38_gdn.cu cuda/q38_ple.cu cuda/cuda_prefix_snapshot_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(filter %.cu,$^) -o $@

cuda-bench: $(CUDA_DECODE_BENCH_BIN) $(CUDA_MOE_BENCH_BIN)
	./$(CUDA_DECODE_BENCH_BIN)
	./$(CUDA_MOE_BENCH_BIN)
	Q38_CUDA_DECODE_MOE=scalar ./$(CUDA_MOE_BENCH_BIN)

.PHONY: moe-route-test
moe-route-test: $(BUILD_DIR)/q38_moe_route_tests
	./$(BUILD_DIR)/q38_moe_route_tests

$(BUILD_DIR)/q38_moe_route_tests: cuda/q38_moe_prefill.cu cuda/q38_moe_route_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(filter %.cu,$^) -o $@

clean:
	rm -rf $(BUILD_DIR)

CUDA_TE_BIND_SMOKE_BIN := $(BUILD_DIR)/q38-te-bind-smoke
.PHONY: te-bind-smoke
te-bind-smoke: $(CUDA_TE_BIND_SMOKE_BIN)
$(CUDA_TE_BIND_SMOKE_BIN): $(CUDA_RUNTIME_SOURCES) cuda/cuda_te_bind_smoke.cc $(RUNTIME_SOURCES) $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $(CUDA_EXTRA_FLAGS) $(CUDA_RUNTIME_SOURCES) cuda/cuda_te_bind_smoke.cc $(RUNTIME_SOURCES) -o $@ $(CUDA_LIBS)


CUDA_TE_DECODE_SMOKE_BIN := $(BUILD_DIR)/q38-te-decode-smoke
.PHONY: te-decode-smoke
te-decode-smoke: $(CUDA_TE_DECODE_SMOKE_BIN)
$(CUDA_TE_DECODE_SMOKE_BIN): $(CUDA_RUNTIME_SOURCES) cuda/cuda_te_decode_smoke.cc $(RUNTIME_SOURCES) $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $(CUDA_EXTRA_FLAGS) $(CUDA_RUNTIME_SOURCES) cuda/cuda_te_decode_smoke.cc $(RUNTIME_SOURCES) -o $@ $(CUDA_LIBS)

CUDA_TE_TEST_BIN := $(BUILD_DIR)/q38-te-tests
.PHONY: te-test
te-test: $(CUDA_TE_TEST_BIN)
	./$(CUDA_TE_TEST_BIN)
$(CUDA_TE_TEST_BIN): cuda/cuda_te.cu cuda/q38_kernels.cu cuda/cuda_te_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: moe-register-test batch-gemv-test
moe-register-test: $(BUILD_DIR)/q38_moe_reg_tests
	./$(BUILD_DIR)/q38_moe_reg_tests

$(BUILD_DIR)/q38_moe_reg_tests: cuda/q38_moe_prefill.cu cuda/q38_moe_reg_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(filter %.cu,$^) -o $@

batch-gemv-test: $(BUILD_DIR)/q38_batch_gemv_tests
	./$(BUILD_DIR)/q38_batch_gemv_tests

$(BUILD_DIR)/q38_batch_gemv_tests: cuda/q38_kernels.cu cuda/q38_batch_gemv_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(filter %.cu,$^) -o $@

.PHONY: native-fast-tests
native-fast-tests: gdn-local-test qsa-local-test mtp-qsa-graph-test head-pipeline-test argmax-test hc-finish-test marlin-tail-test peer-fused-test mtp-input-test

.PHONY: gdn-local-test
gdn-local-test: $(BUILD_DIR)/q38_gdn-local_tests
	./$(BUILD_DIR)/q38_gdn-local_tests
$(BUILD_DIR)/q38_gdn-local_tests: cuda/q38_gdn.cu cuda/q38_gdn_local_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: qsa-local-test
qsa-local-test: $(BUILD_DIR)/q38_qsa-local_tests
	./$(BUILD_DIR)/q38_qsa-local_tests
$(BUILD_DIR)/q38_qsa-local_tests: cuda/q38_qsa.cu cuda/q38_qsa_local_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: mtp-qsa-graph-test
mtp-qsa-graph-test: $(BUILD_DIR)/q38_mtp-qsa-graph_tests
	./$(BUILD_DIR)/q38_mtp-qsa-graph_tests
$(BUILD_DIR)/q38_mtp-qsa-graph_tests: cuda/q38_qsa.cu cuda/q38_qsa_mtp_graph_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: qsa-sparse-score-test
qsa-sparse-score-test: $(BUILD_DIR)/q38_qsa_sparse_score_tests
	./$(BUILD_DIR)/q38_qsa_sparse_score_tests
$(BUILD_DIR)/q38_qsa_sparse_score_tests: cuda/q38_qsa.cu cuda/q38_qsa_sparse_score_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: head-pipeline-test
head-pipeline-test: $(BUILD_DIR)/q38_head-pipeline_tests
	./$(BUILD_DIR)/q38_head-pipeline_tests
$(BUILD_DIR)/q38_head-pipeline_tests: cuda/q38_kernels.cu cuda/q38_hyper.cu cuda/q38_head_pipeline_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: argmax-test
argmax-test: $(BUILD_DIR)/q38_argmax_tests
	./$(BUILD_DIR)/q38_argmax_tests
$(BUILD_DIR)/q38_argmax_tests: cuda/q38_kernels.cu cuda/q38_argmax_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: hc-finish-test marlin-tail-test
hc-finish-test: $(BUILD_DIR)/q38_hc_finish_tests
	./$(BUILD_DIR)/q38_hc_finish_tests
$(BUILD_DIR)/q38_hc_finish_tests: cuda/q38_hyper.cu cuda/q38_moe.cu cuda/q38_hc_finish_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@
marlin-tail-test: $(BUILD_DIR)/q38_marlin_tail_tests
	./$(BUILD_DIR)/q38_marlin_tail_tests
$(BUILD_DIR)/q38_marlin_tail_tests: cuda/q38_marlin.cu cuda/q38_moe_prefill.cu cuda/q38_marlin_tail_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@

.PHONY: peer-fused-test
.PHONY: mtp-input-test
mtp-input-test: $(BUILD_DIR)/q38_mtp_input_tests
	./$(BUILD_DIR)/q38_mtp_input_tests
$(BUILD_DIR)/q38_mtp_input_tests: cuda/q38_kernels.cu cuda/q38_hyper.cu cuda/q38_mtp_input_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@ -lcublas

peer-fused-test: $(BUILD_DIR)/q38_peer_fused_tests
	./$(BUILD_DIR)/q38_peer_fused_tests
$(BUILD_DIR)/q38_peer_fused_tests: cuda/cuda_te.cu cuda/q38_peer_fused_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(filter %.cu,$^) -o $@

.PHONY: rsag-test
rsag-test: $(BUILD_DIR)/q38_rsag_tests
	./$(BUILD_DIR)/q38_rsag_tests
$(BUILD_DIR)/q38_rsag_tests: cuda/cuda_te.cu cuda/q38_rsag_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CUDA_COMPAT_FLAGS) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter %.cu,$^) -o $@ -lcublas

.PHONY: bf16-transport-test
bf16-transport-test: $(BUILD_DIR)/q38_bf16_transport_tests
	./$<
$(BUILD_DIR)/q38_bf16_transport_tests: cuda/cuda_te.cu cuda/q38_bf16_transport_tests.cu $(HEADERS) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_COMPAT_FLAGS) $(filter %.cu,$^) -o $@ -lcublas
