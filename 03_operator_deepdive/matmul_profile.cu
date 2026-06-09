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

__global__ void matmul_naive(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}

__global__ void matmul_tiled(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c,
                             int M, int N, int K) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + 31) / 32; t++) {
        if (row < M && t * 32 + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = a[row * K + t * 32 + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * 32 + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = b[(t * 32 + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

void profile_matmul(int M, int N, int K, int num_runs) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size_a));
    CHECK_CUDA(cudaMalloc(&d_b, size_b));
    CHECK_CUDA(cudaMalloc(&d_c, size_c));
    CHECK_CUDA(cudaMemset(d_a, 1, size_a));
    CHECK_CUDA(cudaMemset(d_b, 1, size_b));
    CHECK_CUDA(cudaMemset(d_c, 0, size_c));

    dim3 block_n(16, 16);
    dim3 grid_n((N + 15) / 16, (M + 15) / 16);

    dim3 block_t(32, 32);
    dim3 grid_t((N + 31) / 32, (M + 31) / 32);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start, 0));
    CHECK_CUDA(cudaEventCreate(&stop, 0));

    double ops = 2.0 * M * N * K;

    matmul_naive<<<grid_n, block_n>>>(d_a, d_b, d_c, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        matmul_naive<<<grid_n, block_n>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_naive = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= num_runs;

    matmul_tiled<<<grid_t, block_t>>>(d_a, d_b, d_c, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        matmul_tiled<<<grid_t, block_t>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tiled = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tiled, start, stop));
    ms_tiled /= num_runs;

    printf("%5dx%-5dx%-5d | %10.3f | %10.3f | %8.2f | %8.2f | %6.2fx\n",
           M, N, K, ms_naive, ms_tiled,
           ops / (ms_naive * 1e6), ops / (ms_tiled * 1e6),
           ms_naive / ms_tiled);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("=== MatMul Shape Scan ===\n\n");
    printf("Shape          | Naive(ms) | Tiled(ms) | GFLOPS(N) | GFLOPS(T) | Speedup\n");
    printf("---------------|-----------|-----------|-----------|-----------|--------\n");

    int shapes[][3] = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {1024, 1024, 4096},
        {4096, 1024, 1024},
        {1024, 4096, 1024},
    };
    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int i = 0; i < num_shapes; i++) {
        profile_matmul(shapes[i][0], shapes[i][1], shapes[i][2], 50);
    }

    printf("\n=== Analysis ===\n");
    printf("1. Small matrices: launch overhead dominates, tiled may not help\n");
    printf("2. Large matrices: tiled shows speedup via shared memory reuse\n");
    printf("3. Non-square shapes reveal memory access pattern bottlenecks\n");
    printf("4. Run: ncu --set full ./matmul_profile  for detailed metrics\n");

    return 0;
}
