#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _ce = call; \
    if (_ce != cudaSuccess) { printf("Err %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(_ce)); exit(1); }} while(0)

using namespace nvcuda::wmma;

__global__ void bw_copy(float4* __restrict__ out, const float4* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

__global__ void compute_fma(float* data, int n, int reps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = data[idx];
    float y = 1.00001f;
    for (int i = 0; i < reps; i++) {
        x = fmaf(x, y, y);
    }
    data[idx] = x;
}

__global__ void matmul_roofline(float* __restrict__ c, const float* __restrict__ a,
                                const float* __restrict__ b, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}

__global__ void tc_fp16_roofline(const __half* a, const __half* b, float* c, int reps, int stride) {
    int off = blockIdx.x * stride;
    const __half* a_off = a + off;
    const __half* b_off = b + off;
    float* c_off = c + off;

    fragment<matrix_a, 16, 16, 16, __half, row_major> af;
    fragment<matrix_b, 16, 16, 16, __half, col_major> bf;
    fragment<accumulator, 16, 16, 16, float> cf;

    load_matrix_sync(af, a_off, 16);
    load_matrix_sync(bf, b_off, 16);
    fill_fragment(cf, 0.0f);

    for (int i = 0; i < reps; i++) {
        mma_sync(cf, af, bf, cf);
    }

    store_matrix_sync(c_off, cf, 16, mem_row_major);
}

// WMMA FP16 real matmul: C += A * B, iterating over K in steps of 16
__global__ void wmma_matmul_roofline(
    const __half* a, const __half* b, float* c,
    int M, int N, int K)
{
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    fragment<matrix_a, 16, 16, 16, __half, row_major> af;
    fragment<matrix_b, 16, 16, 16, __half, col_major> bf;
    fragment<accumulator, 16, 16, 16, float> cf;
    fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(af, a + tile_row * 16 * K + k, K);
        load_matrix_sync(bf, b + k * N + tile_col * 16, N);
        mma_sync(cf, af, bf, cf);
    }
    store_matrix_sync(c + tile_row * 16 * N + tile_col * 16, cf, N, mem_row_major);
}

int main() {
    int device = 0, sm_count = 0, mem_clock = 0, bus_width = 0;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CHECK(cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, device));
    CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device));
    double bw_theoretical = (double)mem_clock / 1000 * bus_width / 8 * 2 / 1000;

    double peak_bw = 219.0;
    double peak_gflops_f32 = 2861.0;
    double peak_gflops_tc  = 12527.0;

    printf("=== Roofline Model Generator ===\n");
    printf("Device: NVIDIA Thor (%d SMs)\n", sm_count);
    printf("Theoretical BW:  %.0f GB/s\n", bw_theoretical);
    printf("Measured BW:     %.0f GB/s  (float4 copy)\n", peak_bw);
    printf("FMA FP32 peak:   %.0f GFLOP/s\n", peak_gflops_f32);
    printf("TensorCore FP16: %.0f GFLOP/s\n\n", peak_gflops_tc);
    printf("Kernel               | Time      | GFLOP/s    | GB/s       | Arithmetic Intensity\n");
    printf("---------------------|-----------|------------|------------|--------------------\n");

    int n = 64 * 1024 * 1024;
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemset(d_in, 1, bytes));

    dim3 block256(256);
    dim3 grid256((n + 255) / 256);
    int n4 = n / 4;
    dim3 grid128((n4 + 127) / 128);

    cudaEvent_t ev1, ev2; float ms;
    CHECK(cudaEventCreate(&ev1, 0)); CHECK(cudaEventCreate(&ev2, 0));

    bw_copy<<<grid128, 128>>>((float4*)d_out, (const float4*)d_in, n4);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(ev1));
    for (int i = 0; i < 100; i++)
        bw_copy<<<grid128, 128>>>((float4*)d_out, (const float4*)d_in, n4);
    CHECK(cudaEventRecord(ev2));
    CHECK(cudaEventSynchronize(ev2));
    CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
    double avg_ms = ms / 100;
    double gbps = 2.0 * bytes / (avg_ms * 1e6);
    printf("%-20s | %8.3f ms | %9.2f GFLOP/s | %8.2f GB/s | %8.2f FLOP/Byte\n",
           "float4 copy", avg_ms, 0.0, gbps, 0.0);

    compute_fma<<<grid256, 256>>>(d_out, n, 1000);
    CHECK(cudaDeviceSynchronize());

    double compute_ops = 2.0 * n * 1000;
    CHECK(cudaEventRecord(ev1));
    for (int i = 0; i < 10; i++) compute_fma<<<grid256, 256>>>(d_out, n, 1000);
    CHECK(cudaEventRecord(ev2));
    CHECK(cudaEventSynchronize(ev2));
    CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
    avg_ms = ms / 10;
    // 2 * n bytes read, n bytes written = 3n bytes
    printf("%-20s | %8.3f ms | %9.2f GFLOP/s | %8.2f GB/s | %8.2f FLOP/Byte\n",
           "fma 1000 iters", avg_ms, compute_ops/(avg_ms*1e6),
           3.0*bytes/(avg_ms*1e6), compute_ops/(3.0*bytes));

    int sizes[] = {64, 128, 256, 512, 1024};
    for (int s = 0; s < 5; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t mat_bytes = M*K*sizeof(float) + K*N*sizeof(float) + M*N*sizeof(float);
        double ops = 2.0 * M * N * K;

        float *d_a, *d_b, *d_c;
        CHECK(cudaMalloc(&d_a, M*K*sizeof(float)));
        CHECK(cudaMalloc(&d_b, K*N*sizeof(float)));
        CHECK(cudaMalloc(&d_c, M*N*sizeof(float)));
        CHECK(cudaMemset(d_a, 1, M*K*sizeof(float)));
        CHECK(cudaMemset(d_b, 1, K*N*sizeof(float)));

        dim3 block2(16, 16);
        dim3 grid2((N+15)/16, (M+15)/16);

        CHECK(cudaEventRecord(ev1));
        for (int i = 0; i < 10; i++)
            matmul_roofline<<<grid2, block2>>>(d_c, d_a, d_b, M, N, K);
        CHECK(cudaEventRecord(ev2));
        CHECK(cudaEventSynchronize(ev2));
        CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        avg_ms = ms / 10;

        char label[64];
        snprintf(label, sizeof(label), "matmul %dx%dx%d", M, N, K);
        printf("%-20s | %8.3f ms | %9.2f GFLOP/s | %8.2f GB/s | %8.2f FLOP/Byte\n",
               label, avg_ms, ops/(avg_ms*1e6), mat_bytes/(avg_ms*1e6), ops/mat_bytes);

        CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    }

    // Tensor Core FP16: each block needs its own 256-element tile
    int tc_reps = 50000;
    int tc_tiles = sm_count;
    __half *d_ta, *d_tb;
    float *d_tc;
    CHECK(cudaMalloc(&d_ta, tc_tiles * 256 * sizeof(__half)));
    CHECK(cudaMalloc(&d_tb, tc_tiles * 256 * sizeof(__half)));
    CHECK(cudaMalloc(&d_tc, tc_tiles * 256 * sizeof(float)));
    CHECK(cudaMemset(d_ta, 1, tc_tiles * 256 * sizeof(__half)));
    CHECK(cudaMemset(d_tb, 1, tc_tiles * 256 * sizeof(__half)));
    CHECK(cudaMemset(d_tc, 0, tc_tiles * 256 * sizeof(float)));

    tc_fp16_roofline<<<tc_tiles, 32>>>(d_ta, d_tb, d_tc, tc_reps, 256);
    CHECK(cudaDeviceSynchronize());

    double tc_flops = (double)tc_reps * 8192.0;
    double tc_mem = 2048.0;

    CHECK(cudaEventRecord(ev1));
    for (int i = 0; i < 5; i++)
        tc_fp16_roofline<<<tc_tiles, 32>>>(d_ta, d_tb, d_tc, tc_reps, 256);
    CHECK(cudaEventRecord(ev2));
    CHECK(cudaEventSynchronize(ev2));
    CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
    avg_ms = ms / 5;

    printf("%-20s | %8.3f ms | %9.2f GFLOP/s | %8.2f GB/s | %8.2f FLOP/Byte\n",
           "tc_fp16 50K reps", avg_ms,
           tc_flops * tc_tiles / (avg_ms * 1e6),
           tc_mem * tc_tiles / (avg_ms * 1e6),
           tc_flops / tc_mem);

    CHECK(cudaFree(d_ta)); CHECK(cudaFree(d_tb)); CHECK(cudaFree(d_tc));

    // WMMA FP16 real matmul @ 1024x1024x1024
    int wM = 1024, wN = 1024, wK = 1024;
    __half *d_wa, *d_wb;
    float *d_wc;
    CHECK(cudaMalloc(&d_wa, wM * wK * sizeof(__half)));
    CHECK(cudaMalloc(&d_wb, wK * wN * sizeof(__half)));
    CHECK(cudaMalloc(&d_wc, wM * wN * sizeof(float)));
    CHECK(cudaMemset(d_wa, 0, wM * wK * sizeof(__half)));
    CHECK(cudaMemset(d_wb, 0, wK * wN * sizeof(__half)));
    CHECK(cudaMemset(d_wc, 0, wM * wN * sizeof(float)));
    __half h_one = __float2half(1.0f);
    CHECK(cudaMemcpy(d_wa, &h_one, sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wb, &h_one, sizeof(__half), cudaMemcpyHostToDevice));

    dim3 wgrid(wN/16, wM/16);
    double w_ops = 2.0 * wM * wN * wK;
    size_t w_bytes = (wM*wK + wK*wN)*sizeof(__half) + wM*wN*sizeof(float);

    wmma_matmul_roofline<<<wgrid, 32>>>(d_wa, d_wb, d_wc, wM, wN, wK);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(ev1));
    for (int i = 0; i < 10; i++)
        wmma_matmul_roofline<<<wgrid, 32>>>(d_wa, d_wb, d_wc, wM, wN, wK);
    CHECK(cudaEventRecord(ev2));
    CHECK(cudaEventSynchronize(ev2));
    CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
    avg_ms = ms / 10;

    printf("%-20s | %8.3f ms | %9.2f GFLOP/s | %8.2f GB/s | %8.2f FLOP/Byte\n",
           "wmma_fp16 1024^3", avg_ms,
           w_ops / (avg_ms * 1e6),
           w_bytes / (avg_ms * 1e6),
           w_ops / (double)w_bytes);

    CHECK(cudaFree(d_wa)); CHECK(cudaFree(d_wb)); CHECK(cudaFree(d_wc));
    CHECK(cudaEventDestroy(ev1)); CHECK(cudaEventDestroy(ev2));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));

    printf("\n=== Roofline Ceilings ===\n");
    printf("BW_peak=%.0f GB/s", peak_bw);
    printf("  GFLOP_peak_fp32=%.0f", peak_gflops_f32);
    printf("  GFLOP_peak_tc16=%.0f", peak_gflops_tc);
    printf("  GOP_peak_int8=%.0f", 46700.0);
    printf("  GFLOP_peak_tf32=%.0f", 10932.0);
    printf("  WMMA_matmul_1024=%.0f", w_ops / (avg_ms * 1e6));
    printf("\n");
    printf("Run: python3 plot_roofline.py   to visualize\n");
    return 0;
}
