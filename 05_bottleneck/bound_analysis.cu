#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int get_attr(int device, cudaDeviceAttr attr) {
    int val;
    CHECK_CUDA(cudaDeviceGetAttribute(&val, attr, device));
    return val;
}

__global__ void memory_bound_kernel(const float* __restrict__ in,
                                    float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

__global__ void compute_bound_kernel(float* data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = data[idx];
    for (int i = 0; i < iters; i++) {
        x = sinf(cosf(x * x + 0.5f));
        x = expf(logf(fabsf(x) + 1e-6f));
    }
    data[idx] = x;
}

__global__ void mid_arith_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = in[idx];
    for (int i = 0; i < 8; i++) {
        x = x * 0.5f + 0.5f;
    }
    out[idx] = x;
}

__global__ void bad_coalescing_kernel(const float* __restrict__ in,
                                      float* __restrict__ out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx * stride];
    }
}

__global__ void low_occupancy_kernel(const float* __restrict__ in,
                                     float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}

double measure(void* kernel, dim3 grid, dim3 block, int smem,
               const float* d_in, float* d_out, int n,
               int num_runs, const char* label,
               bool extra_param = false, int param = 0) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start, 0));
    CHECK_CUDA(cudaEventCreate(&stop, 0));

    for (int i = 0; i < 3; i++) {
        if (kernel == (void*)compute_bound_kernel) {
            compute_bound_kernel<<<grid, block>>>(d_out, n, param);
        } else if (kernel == (void*)bad_coalescing_kernel) {
            bad_coalescing_kernel<<<grid, block>>>(d_in, d_out, n, param);
        } else if (kernel == (void*)memory_bound_kernel) {
            memory_bound_kernel<<<grid, block>>>(d_in, d_out, n);
        } else if (kernel == (void*)mid_arith_kernel) {
            mid_arith_kernel<<<grid, block>>>(d_in, d_out, n);
        } else if (kernel == (void*)low_occupancy_kernel) {
            low_occupancy_kernel<<<grid, block>>>(d_in, d_out, n);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        if (kernel == (void*)compute_bound_kernel) {
            compute_bound_kernel<<<grid, block>>>(d_out, n, param);
        } else if (kernel == (void*)bad_coalescing_kernel) {
            bad_coalescing_kernel<<<grid, block>>>(d_in, d_out, n, param);
        } else if (kernel == (void*)memory_bound_kernel) {
            memory_bound_kernel<<<grid, block>>>(d_in, d_out, n);
        } else if (kernel == (void*)mid_arith_kernel) {
            mid_arith_kernel<<<grid, block>>>(d_in, d_out, n);
        } else if (kernel == (void*)low_occupancy_kernel) {
            low_occupancy_kernel<<<grid, block>>>(d_in, d_out, n);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = ms / num_runs;
    printf("%-30s %8.3f ms\n", label, avg_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return avg_ms;
}

int main() {
    int device = 0;
    int sm_count = get_attr(device, cudaDevAttrMultiProcessorCount);

    printf("=== Bottleneck Classification ===\n");
    printf("Device: NVIDIA Thor, SMs: %d\n\n", sm_count);

    int n = 16 * 1024 * 1024;
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemset(d_in, 1, bytes));

    printf("--- Scenario 1: Memory-Bound Kernel ---\n");
    printf("    Simple read+write per element, no compute reuse\n");
    dim3 block1(256);
    dim3 grid1((n + 255) / 256);
    measure((void*)memory_bound_kernel, grid1, block1, 0,
            d_in, d_out, n, 100, "mem-bound (1 read + 1 write)");

    printf("\n--- Scenario 2: Compute-Bound Kernel ---\n");
    printf("    Heavy math ops, minimal memory access\n");
    measure((void*)compute_bound_kernel, grid1, block1, 0,
            d_in, d_out, n, 20, "compute-bound (100 iters)", true, 100);
    measure((void*)compute_bound_kernel, grid1, block1, 0,
            d_in, d_out, n, 20, "compute-bound (1000 iters)", true, 1000);

    printf("\n--- Scenario 3: Balanced Kernel ---\n");
    printf("    Moderate compute + memory access\n");
    measure((void*)mid_arith_kernel, grid1, block1, 0,
            d_in, d_out, n, 100, "balanced (8 flops per element)");

    printf("\n--- Scenario 4: Bad Memory Access Pattern ---\n");
    printf("    Non-coalesced access due to stride (all process equal elements)\n");
    float *d_big;
    CHECK_CUDA(cudaMalloc(&d_big, bytes * 32));
    CHECK_CUDA(cudaMemset(d_big, 1, bytes * 32));
    measure((void*)bad_coalescing_kernel, grid1, block1, 0,
            d_big, d_out, n, 100, "stride=1 (coalesced)", true, 1);
    measure((void*)bad_coalescing_kernel, grid1, block1, 0,
            d_big, d_out, n, 100, "stride=4 (bad coalescing)", true, 4);
    measure((void*)bad_coalescing_kernel, grid1, block1, 0,
            d_big, d_out, n, 100, "stride=32 (worst)", true, 32);
    CHECK_CUDA(cudaFree(d_big));

    printf("\n--- Scenario 5: Low Occupancy ---\n");
    printf("    Few threads per block limits SM utilization\n");
    dim3 block_low(32);
    dim3 grid_low((n + 31) / 32);
    measure((void*)low_occupancy_kernel, grid_low, block_low, 0,
            d_in, d_out, n, 100, "low occ (32 thr/block)");
    measure((void*)low_occupancy_kernel, grid1, block1, 0,
            d_in, d_out, n, 100, "high occ (256 thr/block)");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    printf("\n=== Bottleneck Diagnosis Guide ===\n");
    printf("Symptom                        | Likely Cause          | Fix\n");
    printf("-------------------------------|-----------------------|-------------------\n");
    printf("SM busy < 30%%                  | Low occupancy         | Increase block size\n");
    printf("Mem pipes busy > 60%%           | Memory-bound          | Tiling / coalescing\n");
    printf("SM busy > 60%%, mem < 30%%       | Compute-bound         | Reduce precision\n");
    printf("Strided access perf drop       | Non-coalesced         | Restructure layout\n");
    printf("Small kernel high overhead     | Launch latency        | CUDA Graph\n");
    printf("Performance << theoretical     | Wrong bound           | Use ncu to check\n");

    return 0;
}
