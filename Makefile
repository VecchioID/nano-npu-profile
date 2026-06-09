NVCC := /usr/local/cuda-13.0/bin/nvcc
ARCH := -arch=sm_110
STD := -std=c++17
NVCC_FLAGS := $(ARCH) $(STD) -O3 -Xcompiler -fopenmp

SRCS := $(wildcard */*.cu)
BINS := $(SRCS:.cu=)

.PHONY: all clean run run-ncu run-nsys

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(BINS)
	rm -rf *.nsys-rep *.ncu-rep

run: all
	@echo "========== 01: Bandwidth Baseline =========="
	./01_baseline_bandwidth/bw_bench
	./01_baseline_bandwidth/compute_bench
	@echo ""
	@echo "========== 02: Roofline =========="
	./02_roofline/roofline_gen
	@echo ""
	@echo "========== 03: Operator Profile =========="
	./03_operator_deepdive/matmul_profile
	./03_operator_deepdive/conv_profile
	@echo ""
	@echo "========== 04: Model Level =========="
	./04_model_level/mini_cnn
	@echo ""
	@echo "========== 05: Bottleneck Analysis =========="
	./05_bottleneck/bound_analysis

run-ncu: all
	@echo "Profiling with ncu..."
	../nano-monitor/profiler.py 2>/dev/null || \
	bash tools/run_ncu.sh ./01_baseline_bandwidth/bw_bench

run-nsys: all
	bash tools/run_nsys.sh ./04_model_level/mini_cnn
