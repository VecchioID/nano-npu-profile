#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void read_bench(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void write_bench(float* __restrict__ data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

__global__ void copy_bench(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

double measure_bw(void (*kernel)(const float*, float*, int),
                  const float* d_in, float* d_out, int n,
                  int num_runs, const char* name) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    size_t bytes = n * sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 3; i++) {
        kernel<<<grid_size, block_size>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        kernel<<<grid_size, block_size>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double avg_ms = ms / num_runs;
    double total_bytes_processed = 2.0 * bytes;
    double bw_gbps = total_bytes_processed / (avg_ms * 1e6);

    printf("%-20s  %8.3f ms  %10.2f GB/s\n", name, avg_ms, bw_gbps);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return bw_gbps;
}

int main() {
    int device = 0;
    int sm_count = get_attr(device, cudaDevAttrMultiProcessorCount);
    int clock_rate = get_attr(device, cudaDevAttrClockRate);
    int mem_clock = get_attr(device, cudaDevAttrMemoryClockRate);
    int bus_width = get_attr(device, cudaDevAttrGlobalMemoryBusWidth);
    size_t total_mem;
    CHECK_CUDA(cudaMemGetInfo(NULL, &total_mem));
    int l2_size = get_attr(device, cudaDevAttrL2CacheSize);

    printf("=== NPU Baseline: Memory Bandwidth ===\n");
    printf("Device: NVIDIA Thor\n");
    printf("SMs: %d\n", sm_count);
    printf("Global Mem: %.0f MB\n", (double)total_mem / (1024*1024));
    printf("L2 Cache: %d KB\n", l2_size / 1024);
    printf("Memory Clock: %.0f MHz\n", (double)mem_clock / 1000);
    printf("Bus Width: %d bits\n\n", bus_width);
    printf("Test                     Avg Time    Bandwidth\n");
    printf("-----------------------------------------------\n");

    size_t sizes[] = {
        1 * 1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s] / sizeof(float);
        size_t bytes = sizes[s];

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, bytes));
        CHECK_CUDA(cudaMalloc(&d_out, bytes));
        CHECK_CUDA(cudaMemset(d_in, 1, bytes));

        printf("\n--- Size: %.1f MB ---\n", (double)bytes / (1024*1024));
        measure_bw(copy_bench, d_in, d_out, n, 100, "read+write");

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
    }

    printf("\n=== Analysis ===\n");
    printf("Pass | Observation | Implication\n");
    printf("-----|-------------|------------\n");
    printf("Small buf | BW increases with size | Fixed launch overhead diluted\n");
    printf("Large buf | BW plateaus | Reached device memory bandwidth ceiling\n");
    printf("Peak BW   | Compare to theoretical | Compute roofline ceiling\n");

    return 0;
}
