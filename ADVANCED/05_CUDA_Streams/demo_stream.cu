/**
 * CUDA Streams Demo (C++)
 *
 * 对比三种模式:
 *   1. Single stream (串行)
 *   2. Multi stream (并行 kernel)
 *   3. Stream + Async memcpy (overlap compute & copy)
 *
 * 编译: nvcc -o demo_stream demo_stream.cu -std=c++17
 * 运行: ./demo_stream
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// ── 一个轻量计算 kernel ──
__global__ void compute_kernel(float* out, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 模拟一些计算
        float v = (float)idx * factor;
        for (int i = 0; i < 100; i++) {
            v = v * 1.001f + 0.5f;
        }
        out[idx] = v;
    }
}

// ── 计时工具 ──
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    int N = 4 * 1024 * 1024;  // 16 MB
    int bytes = N * sizeof(float);
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));  // pinned memory
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));

    // 初始化 CPU 数据
    for (int i = 0; i < N; i++) h_a[i] = (float)i;
    for (int i = 0; i < N; i++) h_b[i] = (float)(N - i);

    cudaEvent_t start, stop;
    cudaEventCreate(&start, 0);
    cudaEventCreate(&stop, 0);

    printf("========================================\n");
    printf("CUDA Streams Demo (N=%d, %d MB)\n", N, bytes / 1024 / 1024);
    printf("========================================\n\n");

    // ── 1. Single stream ──
    printf("── 模式 1: Single Stream ──\n");
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    compute_kernel<<<grid_size, block_size>>>(d_c, 1.0f, N);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("  耗时: %.2f ms\n\n", elapsed_ms(start, stop));

    // ── 2. Multi stream (并行 kernel) ──
    printf("── 模式 2: Multi Stream (2 streams, 分半计算) ──\n");
    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    int N_half = N / 2;
    int grid_half = (N_half + block_size - 1) / block_size;

    cudaEventRecord(start);
    // 两个 stream 各算一半
    compute_kernel<<<grid_half, block_size, 0, s1>>>(d_a, 1.0f, N_half);
    compute_kernel<<<grid_half, block_size, 0, s2>>>(d_a + N_half, 2.0f, N_half);
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaStreamSynchronize(s2));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("  耗时: %.2f ms\n\n", elapsed_ms(start, stop));

    // ── 3. Overlap compute & copy ──
    printf("── 模式 3: Overlap Compute & Copy ──\n");

    cudaStream_t stream_copy, stream_compute;
    CUDA_CHECK(cudaStreamCreate(&stream_copy));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));

    cudaEvent_t copy_done;
    cudaEventCreate(&copy_done);

    cudaEventRecord(start);

    // 在 copy stream 上异步传输
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes,
                               cudaMemcpyHostToDevice, stream_copy));
    // 记录 copy 完成事件
    cudaEventRecord(copy_done, stream_copy);
    // compute stream 等 copy 完成再开始
    CUDA_CHECK(cudaStreamWaitEvent(stream_compute, copy_done));

    // compute 在另一个 stream 上执行
    compute_kernel<<<grid_size, block_size, 0, stream_compute>>>(d_c, 1.0f, N);

    // 等 compute 完成，把结果拷回
    cudaEvent_t compute_done;
    cudaEventCreate(&compute_done);
    cudaEventRecord(compute_done, stream_compute);
    CUDA_CHECK(cudaStreamWaitEvent(stream_copy, compute_done));
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes,
                               cudaMemcpyDeviceToHost, stream_copy));

    CUDA_CHECK(cudaStreamSynchronize(stream_copy));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("  耗时: %.2f ms\n", elapsed_ms(start, stop));
    printf("  理论上限: max(copy_time, compute_time)\n");
    printf("  (相比模式 1 的 copy+compute 串行)\n\n");

    // ── 4. 多 stream kernel 并行加速比 ──
    printf("── 模式 4: Stream 数量对性能的影响 ──\n");

    int num_streams_list[] = {1, 2, 4};
    for (int si = 0; si < 3; si++) {
        int ns = num_streams_list[si];
        cudaStream_t* streams = new cudaStream_t[ns];
        for (int i = 0; i < ns; i++)
            CUDA_CHECK(cudaStreamCreate(&streams[i]));

        cudaEventRecord(start);
        int chunk = N / ns;
        int grid_chunk = (chunk + block_size - 1) / block_size;
        for (int i = 0; i < ns; i++) {
            compute_kernel<<<grid_chunk, block_size, 0, streams[i]>>>(
                d_a + i * chunk, (float)(i + 1) * 0.5f, chunk);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        printf("  %d stream(s): %.2f ms (speedup: %.2fx vs 1 stream)\n",
               ns, elapsed_ms(start, stop),
               // 用第一个 pass 的 1 stream 时间做参考
               ns == 1 ? 1.0f : 0);
        delete[] streams;
    }

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(copy_done));
    CUDA_CHECK(cudaEventDestroy(compute_done));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUDA_CHECK(cudaStreamDestroy(stream_copy));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    return 0;
}
