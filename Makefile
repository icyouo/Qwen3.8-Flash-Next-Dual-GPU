CXX ?= c++
PYTHON ?= python3
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -Iinclude

CUDA_HOME ?= /usr/local/cuda-13.1
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOST_CXX ?= g++-14
# The production CMP 170HX boards report GA100 compute capability 8.0.
# Keep this overridable for development hosts with a different CUDA target.
CUDA_ARCH ?= sm_80
NVCCFLAGS ?= -O3 -lineinfo -std=c++17 -arch=$(CUDA_ARCH)
CUDA_EXTRA_FLAGS ?=
CUDA_LIBS ?= -lcublas

BUILD_DIR := build
RUNTIME_SOURCES := src/contracts.cc src/identity.cc src/snapshot.cc src/metrics.cc src/artifact.cc src/tensor_index.cc src/mapped_weights.cc src/device_artifact.cc src/model_plan.cc src/ple.cc src/ple_backend.cc src/sampling.cc src/transaction.cc src/mock_backend.cc src/executor.cc src/rpc.cc
TEST_BIN := $(BUILD_DIR)/q38_runtime_tests
CLI_BIN := $(BUILD_DIR)/q38-runtime
CUDA_CHECK_BIN := $(BUILD_DIR)/q38_cuda_compile_check
CUDA_RUNTIME_BIN := $(BUILD_DIR)/q38-cuda-runtime
CUDA_TEST_BIN := $(BUILD_DIR)/q38_cuda_kernel_tests
CUDA_DECODE_BENCH_BIN := $(BUILD_DIR)/q38_decode_bench
CUDA_MOE_BENCH_BIN := $(BUILD_DIR)/q38_moe_bench
CUDA_COMPAT_STAMP := $(BUILD_DIR)/cuda-compat/.stamp
CUDA_RUNTIME_SOURCES := cuda/cuda_transport.cu cuda/cuda_weights.cu cuda/q38_kernels.cu cuda/q38_gdn.cu cuda/q38_qsa.cu cuda/q38_hyper.cu cuda/q38_moe.cu cuda/q38_ple.cu cuda/cuda_backend.cu

.PHONY: all test python-test verify cuda-check cuda-runtime cuda-test cuda-bench strict-gate clean

all: test

$(BUILD_DIR):
	mkdir -p $@

$(TEST_BIN): $(RUNTIME_SOURCES) tests/test_runtime.cc | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(CLI_BIN): $(RUNTIME_SOURCES) src/main.cc | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(CUDA_COMPAT_STAMP): tools/q38_cuda_compat.py | $(BUILD_DIR)
	$(PYTHON) $< \
		--source-root $(CUDA_HOME)/targets/x86_64-linux/include \
		--output-root $(BUILD_DIR)/cuda-compat

$(CUDA_CHECK_BIN): $(CUDA_RUNTIME_SOURCES) cuda/cuda_compile_check.cc $(RUNTIME_SOURCES) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) -I$(BUILD_DIR)/cuda-compat $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter-out $(CUDA_COMPAT_STAMP),$^) -o $@ $(CUDA_LIBS)

$(CUDA_RUNTIME_BIN): $(CUDA_RUNTIME_SOURCES) cuda/cuda_main.cc $(RUNTIME_SOURCES) $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) -I$(BUILD_DIR)/cuda-compat $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter-out $(CUDA_COMPAT_STAMP),$^) -o $@ $(CUDA_LIBS)

$(CUDA_TEST_BIN): cuda/cuda_transport.cu cuda/q38_kernels.cu cuda/q38_gdn.cu cuda/q38_qsa.cu cuda/q38_moe.cu cuda/q38_ple.cu cuda/cuda_kernel_tests.cu src/contracts.cc $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) -I$(BUILD_DIR)/cuda-compat $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter-out $(CUDA_COMPAT_STAMP),$^) -o $@

$(CUDA_DECODE_BENCH_BIN): cuda/q38_kernels.cu cuda/q38_decode_bench.cu $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) -I$(BUILD_DIR)/cuda-compat $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter-out $(CUDA_COMPAT_STAMP),$^) -o $@

$(CUDA_MOE_BENCH_BIN): cuda/q38_moe.cu cuda/q38_moe_bench.cu $(CUDA_COMPAT_STAMP) | $(BUILD_DIR)
	$(NVCC) -ccbin $(CUDA_HOST_CXX) -I$(BUILD_DIR)/cuda-compat $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_EXTRA_FLAGS) $(filter-out $(CUDA_COMPAT_STAMP),$^) -o $@

test: $(TEST_BIN) $(CLI_BIN)
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

cuda-bench: $(CUDA_DECODE_BENCH_BIN) $(CUDA_MOE_BENCH_BIN)
	./$(CUDA_DECODE_BENCH_BIN)
	./$(CUDA_MOE_BENCH_BIN)
	Q38_CUDA_DECODE_MOE=scalar ./$(CUDA_MOE_BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)
