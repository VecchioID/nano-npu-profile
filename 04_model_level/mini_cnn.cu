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

__global__ void conv_layer(const float* __restrict__ input,
                           const float* __restrict__ weight,
                           const float* __restrict__ bias,
                           float* __restrict__ output,
                           int H, int W, int C, int K, int R, int S) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (x >= W || y >= H || k >= K) return;

    float sum = bias[k];
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int ix = x + s;
                int iy = y + r;
                if (ix < W && iy < H) {
                    sum += input[iy * W * C + ix * C + c] *
                           weight[k * C * R * S + c * R * S + r * S + s];
                }
            }
        }
    }
    output[k * H * W + y * W + x] = sum;
}

__global__ void relu_layer(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void maxpool_layer(const float* __restrict__ input,
                              float* __restrict__ output,
                              int H, int W, int C, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    int out_H = H / pool_size;
    int out_W = W / pool_size;
    if (x >= out_W || y >= out_H || c >= C) return;

    float max_val = -1e10f;
    for (int r = 0; r < pool_size; r++) {
        for (int s = 0; s < pool_size; s++) {
            float val = input[(y * pool_size + r) * W * C + (x * pool_size + s) * C + c];
            if (val > max_val) max_val = val;
        }
    }
    output[c * out_H * out_W + y * out_W + x] = max_val;
}

__global__ void fc_layer(const float* __restrict__ input,
                         const float* __restrict__ weight,
                         const float* __restrict__ bias,
                         float* __restrict__ output,
                         int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim) return;

    float sum = bias[idx];
    for (int i = 0; i < in_dim; i++) {
        sum += input[i] * weight[idx * in_dim + i];
    }
    output[idx] = sum;
}

int main() {
    int device = 0;
    int sm_count = get_attr(device, cudaDevAttrMultiProcessorCount);
    int clock_rate = get_attr(device, cudaDevAttrClockRate);

    printf("=== Mini CNN End-to-End Profile ===\n");
    printf("Device: NVIDIA Thor, SMs: %d, Clock: %.0f MHz\n", sm_count, (double)clock_rate / 1000);
    printf("Architecture: Conv(3x3) -> ReLU -> MaxPool(2x2) -> Conv(3x3) -> ReLU -> FC\n\n");

    int H = 32, W = 32, C = 3;
    int K1 = 16, K2 = 32;
    int pool_size = 2;

    float *d_input, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    size_t input_size = H * W * C * sizeof(float);
    size_t w1_size = K1 * C * 3 * 3 * sizeof(float);
    size_t b1_size = K1 * sizeof(float);
    size_t w2_size = K2 * K1 * 3 * 3 * sizeof(float);
    size_t b2_size = K2 * sizeof(float);
    int H2 = H / pool_size, W2 = W / pool_size;
    size_t l1_out_size = K1 * H * W * sizeof(float);
    size_t l2_out_size = K2 * H2 * W2 * sizeof(float);
    int fc_in = K2 * H2 * W2;
    int fc_out = 10;
    size_t w3_size = fc_out * fc_in * sizeof(float);
    size_t b3_size = fc_out * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_w1, w1_size));
    CHECK_CUDA(cudaMalloc(&d_b1, b1_size));
    CHECK_CUDA(cudaMalloc(&d_w2, w2_size));
    CHECK_CUDA(cudaMalloc(&d_b2, b2_size));
    CHECK_CUDA(cudaMalloc(&d_w3, w3_size));
    CHECK_CUDA(cudaMalloc(&d_b3, b3_size));

    float *d_l1_out, *d_pool_out, *d_l2_out, *d_fc_out;
    CHECK_CUDA(cudaMalloc(&d_l1_out, l1_out_size));
    CHECK_CUDA(cudaMalloc(&d_pool_out, K1 * H2 * W2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l2_out, l2_out_size));
    CHECK_CUDA(cudaMalloc(&d_fc_out, fc_out * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_input, 1, input_size));
    CHECK_CUDA(cudaMemset(d_w1, 1, w1_size));
    CHECK_CUDA(cudaMemset(d_b1, 0, b1_size));
    CHECK_CUDA(cudaMemset(d_w2, 1, w2_size));
    CHECK_CUDA(cudaMemset(d_b2, 0, b2_size));
    CHECK_CUDA(cudaMemset(d_w3, 1, w3_size));
    CHECK_CUDA(cudaMemset(d_b3, 0, b3_size));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start, 0));
    CHECK_CUDA(cudaEventCreate(&stop, 0));

    int total_runs = 100;
    CHECK_CUDA(cudaEventRecord(start));

    for (int iter = 0; iter < total_runs; iter++) {
        dim3 block_c(16, 16);
        dim3 grid_c((W + 15) / 16, (H + 15) / 16, K1);
        conv_layer<<<grid_c, block_c>>>(d_input, d_w1, d_b1, d_l1_out, H, W, C, K1, 3, 3);

        int l1_elems = K1 * H * W;
        dim3 block_r(256);
        dim3 grid_r((l1_elems + 255) / 256);
        relu_layer<<<grid_r, block_r>>>(d_l1_out, l1_elems);

        dim3 block_p(16, 16);
        dim3 grid_p((W2 + 15) / 16, (H2 + 15) / 16, K1);
        maxpool_layer<<<grid_p, block_p>>>(d_l1_out, d_pool_out, H, W, K1, pool_size);

        dim3 grid_c2((W2 + 15) / 16, (H2 + 15) / 16, K2);
        conv_layer<<<grid_c2, block_c>>>(d_pool_out, d_w2, d_b2, d_l2_out, H2, W2, K1, K2, 3, 3);

        int l2_elems = K2 * H2 * W2;
        dim3 grid_r2((l2_elems + 255) / 256);
        relu_layer<<<grid_r2, block_r>>>(d_l2_out, l2_elems);

        dim3 block_f(256);
        dim3 grid_f((fc_out + 255) / 256);
        fc_layer<<<grid_f, block_f>>>(d_l2_out, d_w3, d_b3, d_fc_out, fc_in, fc_out);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    printf("=== Results ===\n");
    printf("Total inferences: %d\n", total_runs);
    printf("Total time: %.2f ms\n", total_ms);
    printf("Avg time per inference: %.3f ms\n", total_ms / total_runs);
    printf("Throughput: %.1f inferences/sec\n", total_runs / (total_ms / 1000.0));

    double total_ops = total_runs * (
        2.0 * H * W * C * K1 * 3 * 3 +
        K1 * H * W +
        K1 * H2 * W2 * pool_size * pool_size +
        2.0 * H2 * W2 * K1 * K2 * 3 * 3 +
        K2 * H2 * W2 +
        2.0 * fc_out * fc_in
    );
    printf("Estimated GFLOP/s: %.2f\n", total_ops / (total_ms * 1e6));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_w1)); CHECK_CUDA(cudaFree(d_b1));
    CHECK_CUDA(cudaFree(d_w2)); CHECK_CUDA(cudaFree(d_b2));
    CHECK_CUDA(cudaFree(d_w3)); CHECK_CUDA(cudaFree(d_b3));
    CHECK_CUDA(cudaFree(d_l1_out));
    CHECK_CUDA(cudaFree(d_pool_out));
    CHECK_CUDA(cudaFree(d_l2_out));
    CHECK_CUDA(cudaFree(d_fc_out));

    printf("\n=== Profiling Guide ===\n");
    printf("1. Identify slowest layer: check where time is spent\n");
    printf("2. Profile with nsys: bash tools/run_nsys.sh ./mini_cnn\n");
    printf("3. Profile with ncu:  bash tools/run_ncu.sh ./mini_cnn\n");
    printf("4. Key metrics: occupancy, SM busy, mem busy, arithmetic intensity\n");
    printf("5. Optimization: shared memory tiling, Tensor Cores, fused kernels\n");

    return 0;
}
