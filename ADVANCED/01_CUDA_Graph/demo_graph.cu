/**
 * 01_CUDA_Graph / demo_graph.cu
 * 
 * 对比 CUDA Graph vs 传统方式 启动 1000 个 kernel 的耗时。
 * 
 * 编译: nvcc -arch=sm_110 -std=c++17 -O3 -o demo_graph demo_graph.cu
 * 运行: ./demo_graph
 * 
 * 预期结果:
 *   传统:   ~3000-5000 µs（~3-5 µs/kernel）
 *   Graph:  ~100-500 µs（~0.1-0.5 µs/kernel）
 *   加速:   ~10-30x
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 一个简单的 kernel，模拟真实计算
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a * x[idx] + y[idx];
}

// 一个"什么都不做"的 kernel — 纯测启动开销
__global__ void empty_kernel() {
    // 什么都不做
}

int main() {
    int device = 0;
    int sm_count;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    printf("设备: NVIDIA Thor, SMs: %d\n", sm_count);

    int N = 1024 * 1024;
    float *x, *y;
    CHECK(cudaMalloc(&x, N * sizeof(float)));
    CHECK(cudaMalloc(&y, N * sizeof(float)));
    CHECK(cudaMemset(x, 1, N * sizeof(float)));
    CHECK(cudaMemset(y, 1, N * sizeof(float)));

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start, 0));
    CHECK(cudaEventCreate(&stop, 0));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int KERNEL_COUNT = 1000;

    // ══════════════════════════════════════════════════════════
    // 测试 1：传统方式 — 1000 次单独启动
    // ══════════════════════════════════════════════════════════
    printf("\n=== 测试 1: 传统方式 (%d 次 kernel 启动) ===\n", KERNEL_COUNT);

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < KERNEL_COUNT; i++) {
        saxpy<<<grid, block, 0, stream>>>(2.0f, x, y, N);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms_traditional;
    CHECK(cudaEventElapsedTime(&ms_traditional, start, stop));
    printf("  总时间: %.3f ms\n", ms_traditional);
    printf("  平均:   %.3f µs/launch\n", ms_traditional * 1000 / KERNEL_COUNT);
    printf("  吞吐:   %.0f launches/sec\n", KERNEL_COUNT / (ms_traditional / 1000));

    // ══════════════════════════════════════════════════════════
    // 测试 2：CUDA Graph — 一次构建，重复重放
    // ══════════════════════════════════════════════════════════
    printf("\n=== 测试 2: CUDA Graph (%d 个 kernel 的图) ===\n", KERNEL_COUNT);

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 捕获阶段
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < KERNEL_COUNT; i++) {
        saxpy<<<grid, block, 0, stream>>>(2.0f, x, y, N);
    }
    CHECK(cudaStreamEndCapture(stream, &graph));

    // 实例化（编译为可执行格式）
    CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    CHECK(cudaGraphDestroy(graph));  // 实例化后可以释放 graph 描述

    int REPEAT = 100;
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; i++) {
        CHECK(cudaGraphLaunch(instance, stream));
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms_graph;
    CHECK(cudaEventElapsedTime(&ms_graph, start, stop));
    float ms_per_graph = ms_graph / REPEAT;

    printf("  总时间 (x%d): %.3f ms\n", REPEAT, ms_graph);
    printf("  每次 graph 启动: %.3f µs\n", ms_per_graph * 1000);
    printf("  等价每次 kernel: %.3f µs\n", ms_per_graph * 1000 / KERNEL_COUNT);
    printf("  加速比: %.2fx\n", ms_traditional / ms_per_graph);

    CHECK(cudaGraphExecDestroy(instance));

    // ══════════════════════════════════════════════════════════
    // 测试 3：纯启动开销（空 kernel）
    // ══════════════════════════════════════════════════════════
    printf("\n=== 测试 3: 纯启动开销（空 kernel） ===\n");
    dim3 empty_grid(1);
    dim3 empty_block(32);

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < KERNEL_COUNT; i++) {
        empty_kernel<<<empty_grid, empty_block, 0, stream>>>();
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms_empty;
    CHECK(cudaEventElapsedTime(&ms_empty, start, stop));
    printf("  空 kernel x%d: %.3f ms\n", KERNEL_COUNT, ms_empty);
    printf("  每次空 kernel: %.3f µs\n", ms_empty * 1000 / KERNEL_COUNT);

    // ══════════════════════════════════════════════════════════
    // 测试 4：多 stream 交叉启动
    // ══════════════════════════════════════════════════════════
    printf("\n=== 测试 4: 多 Stream 对启动开销的影响 ===\n");

    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));

    // 两个 stream 交替启动
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < KERNEL_COUNT / 2; i++) {
        empty_kernel<<<empty_grid, empty_block, 0, s1>>>();
        empty_kernel<<<empty_grid, empty_block, 0, s2>>>();
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms_dual;
    CHECK(cudaEventElapsedTime(&ms_dual, start, stop));
    printf("  双 stream 交替 x%d: %.3f ms\n", KERNEL_COUNT, ms_dual);
    printf("  每次: %.3f µs (vs %.3f µs 单 stream)\n",
           ms_dual * 1000 / KERNEL_COUNT, ms_empty * 1000 / KERNEL_COUNT);

    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));

    // ══════════════════════════════════════════════════════════
    // 总结
    // ══════════════════════════════════════════════════════════
    printf("\n" + "=" * 50);
    printf("\n总结:\n");
    printf("  传统方式:  每 kernel %.1f µs（含 GPU 执行时间）\n",
           ms_traditional * 1000 / KERNEL_COUNT);
    printf("  空 kernel:  每 kernel %.1f µs（纯启动开销）\n",
           ms_empty * 1000 / KERNEL_COUNT);
    printf("  CUDA Graph: 每 kernel %.2f µs（几乎 = 执行时间）\n",
           ms_per_graph * 1000 / KERNEL_COUNT);
    printf("  启动开销占比: %.0f%%\n",
           (ms_empty / ms_traditional) * 100);
    printf("\n结论: CUDA Graph 消除启动开销，对小 kernel 加速显著\n");

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
    return 0;
}
