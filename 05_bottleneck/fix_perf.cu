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

__global__ void bad_reduce(const float* __restrict__ in,
                           float* __restrict__ out, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[256];
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 256; i++) {
            sum += sdata[i];
        }
        out[blockIdx.x] = sum;
    }
}

__global__ void good_reduce(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[256];
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

__global__ void warp_reduce(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? in[idx] : 0.0f;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_sum[8];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) warp_sum[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (tid < 8) ? warp_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (tid == 0) out[blockIdx.x] = val;
    }
}

int main() {
    printf("=== From Profiling to Optimization ===\n\n");

    int n = 64 * 1024 * 1024;
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, n / 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_in, 1, bytes));

    dim3 block(256);
    dim3 grid(n / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start, 0));
    CHECK_CUDA(cudaEventCreate(&stop, 0));

    printf("--- Case: Reduction Optimization ---\n");
    printf("Array: %d floats (%.0f MB)\n", n, (double)bytes / (1024*1024));
    printf("\n");

    for (int i = 0; i < 3; i++) {
        bad_reduce<<<grid, block>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        bad_reduce<<<grid, block>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_bad = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_bad, start, stop));
    printf("Bad  (serial reduction):  %8.3f ms  (100 runs)\n", ms_bad / 100);

    for (int i = 0; i < 3; i++) {
        good_reduce<<<grid, block>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        good_reduce<<<grid, block>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_good = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_good, start, stop));
    printf("Good (tree reduction):    %8.3f ms  (100 runs)\n", ms_good / 100);

    for (int i = 0; i < 3; i++) {
        warp_reduce<<<grid, block>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        warp_reduce<<<grid, block>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_warp = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_warp, start, stop));
    printf("Warp (shuffle reduction): %8.3f ms  (100 runs)\n", ms_warp / 100);

    printf("\n--- Speedup ---\n");
    printf("Tree reduction vs Serial:  %.2fx\n", ms_bad / ms_good);
    printf("Warp shuffle vs Serial:    %.2fx\n", ms_bad / ms_warp);

    printf("\n=== Optimization Workflow ===\n");
    printf("1. Profile (ncu) → identify bottleneck\n");
    printf("2. Diagnose → check occupancy, memory pattern, compute intensity\n");
    printf("3. Fix → apply pattern (tiling, coalescing, shuffle, etc.)\n");
    printf("4. Re-profile → verify improvement\n");
    printf("5. Iterate → until bottleneck shifts or meets target\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
