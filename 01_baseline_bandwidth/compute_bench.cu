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

__global__ void compute_heavy(float* data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = data[idx];
    for (int i = 0; i < iters; i++) {
        x = x * x + 0.5f;
        x = sinf(x) * cosf(x);
    }
    data[idx] = x;
}

__global__ void mac_bench(float* __restrict__ a, const float* __restrict__ b,
                          const float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < 256; i++) {
        acc += b[idx] * c[idx];
    }
    a[idx] = acc;
}

int main() {
    int device = 0;
    int sm_count = get_attr(device, cudaDevAttrMultiProcessorCount);
    int max_threads = get_attr(device, cudaDevAttrMaxThreadsPerMultiProcessor);
    int clock_rate = get_attr(device, cudaDevAttrClockRate);

    printf("=== NPU Baseline: Compute Throughput ===\n");
    printf("Device: NVIDIA Thor\n");
    printf("SM Count: %d\n", sm_count);
    printf("Max Threads/SM: %d\n", max_threads);
    printf("Clock Rate: %.0f MHz\n\n", (double)clock_rate / 1000);

    int n = 4 * 1024 * 1024;
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMemset(d_a, 0, bytes));
    CHECK_CUDA(cudaMemset(d_b, 1, bytes));
    CHECK_CUDA(cudaMemset(d_c, 2, bytes));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    printf("--- MAC Throughput ---\n");
    float ms;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iters[] = {10, 100, 1000};
    for (int i = 0; i < 3; i++) {
        int num_iters = iters[i];
        CHECK_CUDA(cudaEventRecord(start));
        compute_heavy<<<grid_size, block_size>>>(d_a, n, num_iters);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        double flops = (double)n * num_iters * 2.0;
        double gflops = flops / (ms * 1e6);
        printf("iters=%4d  time=%8.3f ms  %10.2f GFLOP/s\n", num_iters, ms, gflops);
    }

    printf("\n--- FMA Throughput ---\n");
    CHECK_CUDA(cudaEventRecord(start));
    mac_bench<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = (double)n * 256 * 2.0;
    double gflops = flops / (ms * 1e6);
    printf("FMA(256) time=%8.3f ms  %10.2f GFLOP/s\n", ms, gflops);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\n=== How to interpret ===\n");
    printf("Compare GFLOP/s vs theoretical peak.\n");
    printf("If << peak: likely memory-bound or occupancy-limited.\n");
    printf("Use: ncu --set full ./compute_bench  to get SM utilization.\n");

    return 0;
}
