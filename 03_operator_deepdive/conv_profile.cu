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

__global__ void conv2d_naive(const float* __restrict__ input,
                             const float* __restrict__ weight,
                             float* __restrict__ output,
                             int H, int W, int C, int K, int R, int S) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_k = blockIdx.z;

    if (out_x >= W || out_y >= H || out_k >= K) return;

    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int in_x = out_x + s;
                int in_y = out_y + r;
                if (in_x < W && in_y < H) {
                    sum += input[in_y * W * C + in_x * C + c] *
                           weight[out_k * C * R * S + c * R * S + r * S + s];
                }
            }
        }
    }
    output[out_k * H * W + out_y * W + out_x] = sum;
}

__global__ void conv2d_implicit_preload(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        int H, int W, int C, int K, int R, int S) {
    extern __shared__ float cache[];
    float* smem_input = cache;
    float* smem_weight = &cache[R * S * C];

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_k = blockIdx.z;

    if (out_x >= W || out_y >= H || out_k >= K) return;

    if (threadIdx.x < R * S && threadIdx.y < C && threadIdx.z == 0) {
        int rs = threadIdx.x;
        int c = threadIdx.y;
        int load_x = out_x + (rs % S);
        int load_y = out_y + (rs / S);
        if (load_x < W && load_y < H) {
            cache[rs * C + c] = input[load_y * W * C + load_x * C + c];
        } else {
            cache[rs * C + c] = 0.0f;
        }
    }

    if (threadIdx.x < R * S && threadIdx.y < C && threadIdx.z == 1) {
        int rs = threadIdx.x;
        int c = threadIdx.y;
        smem_weight[rs * C + c] = weight[out_k * C * R * S + c * R * S + rs];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < R * S * C; i++) {
        sum += cache[i] * smem_weight[i];
    }
    output[out_k * H * W + out_y * W + out_x] = sum;
}

double measure_conv(void* kernel, dim3 grid, dim3 block, int shared_mem,
                    const float* input, const float* weight, float* output,
                    int H, int W, int C, int K, int R, int S,
                    int num_runs, double ops, const char* label) {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start, 0));
    CHECK_CUDA(cudaEventCreate(&stop, 0));

    for (int i = 0; i < 3; i++) {
        if (kernel == (void*)conv2d_naive) {
            conv2d_naive<<<grid, block>>>(input, weight, output, H, W, C, K, R, S);
        } else {
            conv2d_implicit_preload<<<grid, block, shared_mem>>>(
                input, weight, output, H, W, C, K, R, S);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        if (kernel == (void*)conv2d_naive) {
            conv2d_naive<<<grid, block>>>(input, weight, output, H, W, C, K, R, S);
        } else {
            conv2d_implicit_preload<<<grid, block, shared_mem>>>(
                input, weight, output, H, W, C, K, R, S);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = ms / num_runs;
    double gflops = ops / (avg_ms * 1e6);

    printf("%-20s | %8.3f ms | %10.2f GFLOP/s\n", label, avg_ms, gflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return gflops;
}

void run_conv_profile(int H, int W, int C, int K, int R, int S) {
    size_t input_size = H * W * C * sizeof(float);
    size_t weight_size = K * C * R * S * sizeof(float);
    size_t output_size = K * H * W * sizeof(float);

    float *d_input, *d_weight, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));
    CHECK_CUDA(cudaMemset(d_input, 1, input_size));
    CHECK_CUDA(cudaMemset(d_weight, 1, weight_size));
    CHECK_CUDA(cudaMemset(d_output, 0, output_size));

    dim3 block_n(16, 16);
    dim3 grid_n((W + 15) / 16, (H + 15) / 16, K);

    dim3 block_p(8, 8);
    dim3 grid_p((W + 7) / 8, (H + 7) / 8, K);
    int shared_mem = (R * S * C + K * C * R * S) * sizeof(float);

    double ops = 2.0 * H * W * K * C * R * S;

    printf("\n--- Conv: H=%d W=%d C=%d K=%d R=%d S=%d ---\n", H, W, C, K, R, S);
    measure_conv((void*)conv2d_naive, grid_n, block_n, 0,
                 d_input, d_weight, d_output, H, W, C, K, R, S, 20, ops, "naive");
    measure_conv((void*)conv2d_implicit_preload, grid_p, block_p, shared_mem,
                 d_input, d_weight, d_output, H, W, C, K, R, S, 20, ops, "preload");

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_output));
}

int main() {
    printf("=== Convolution 2D Profiling ===\n\n");
    printf("Kernel               | Time      | GFLOP/s\n");
    printf("---------------------|-----------|----------\n");

    run_conv_profile(32, 32, 3, 16, 3, 3);
    run_conv_profile(64, 64, 16, 32, 3, 3);
    run_conv_profile(128, 128, 32, 64, 3, 3);
    run_conv_profile(32, 32, 64, 128, 3, 3);

    printf("\n=== Analysis ===\n");
    printf("1. Small spatial (32x32): naive may be enough, occupancy is key\n");
    printf("2. Large spatial (128x128): preload helps with data reuse\n");
    printf("3. Many channels (C=64,K=128): compute-bound, Tensor Cores matter\n");
    printf("4. Use ncu to check: arithmetic intensity, L1 hit rate, occupancy\n");

    return 0;
}
