# CUDA 从零开始教程

> 目标：学完这个教程，你能读懂 nano-npu-profile 里所有代码，理解每一步在做什么。

---

## 第一章：GPU 凭什么快？

### 1.1 CPU vs GPU

```
CPU： 几个强大的核心（4-16核），每个核极快，适合复杂逻辑
       专为"低延迟"设计

GPU： 几千个弱小的核心，每个核很慢，但数量极多
       专为"高吞吐"设计
```

例子：
- CPU 像 4 个博士生，能解微分方程
- GPU 像 5000 个小学生，每人只做 1+1，但 5000 人同时做

**GPU 适合：大量数据做同样简单的运算（矩阵乘、图像处理）**
**CPU 适合：复杂逻辑、分支多、数据量小**

### 1.2 为什么需要 CUDA？

NVIDIA 的 GPU 不能直接用 C++ 编程。CUDA 是 NVIDIA 提供的扩展，让你写 C++ 函数在 GPU 上执行。

```
普通 C++ 函数 → 在 CPU 上跑
CUDA kernel   → 在 GPU 上跑（用 <<<>>> 启动）
```

---

## 第二章：GPU 内部长什么样？

### 2.1 硬件结构

```
GPU 芯片
  ├── SM 0（Streaming Multiprocessor，流多处理器）
  │     ├── CUDA Core × N（算术逻辑单元）
  │     ├── Shared Memory（共享内存，几十 KB）
  │     ├── Register File（寄存器）
  │     └── Warp Scheduler（调度器）
  ├── SM 1
  ├── ...
  └── SM 19（Thor 有 20 个 SM）

  ── L2 Cache（所有 SM 共享）
  ── Global Memory / VRAM（显存，几十 GB）
```

### 2.2 Warp（线程束）—— 最重要的概念

**Warp = 32 个线程一组，硬件一次执行一条指令，32 个线程同时做。**

```
一个 warp 的 32 个线程：
  Thread 0, Thread 1, ..., Thread 31

这 32 个线程在同一时刻执行完全相同的指令。
如果它们访问不同的数据 → 完美并行（SIMT）
如果它们走不同的分支（if/else）→ warp divergence，串行执行
```

**Warp 是理解 GPU 性能的关键。** 所有优化最终都围绕：如何让 warp 高效工作。

### 2.3 SM（Streaming Multiprocessor）

SM 是真正的计算单元。每个 SM 可以同时运行多个 warp。

```
SM 内有：
  - Warp Scheduler：决定下一个执行哪个 warp
  - CUDA Cores：执行算术指令
  - Shared Memory：block 内线程共享的快速内存
  - Registers：每个线程私有

SM 的工作方式：
  当一个 warp 在等数据（访存延迟 ~200-800 cycles），
  scheduler 立刻切换到另一个 warp → 用计算隐藏延迟
```

**这就是为什么需要很多 warp：warp 越多，SM 越不容易空闲。**

---

## 第三章：CUDA 编程模型（软件）

### 3.1 三层结构

```
你写的 CUDA 代码              GPU 执行
────────────────────          ──────────

Grid（一组 block）             全部 SM 共同完成
  │
  ├── Block 0                    → 分配到一个 SM
  │     ├── Thread 0
  │     ├── Thread 1
  │     ├── ...                 → 分成 warp（每组 32 个）
  │     └── Thread 255          ← 256 线程 = 8 个 warp
  │
  ├── Block 1                    → 分配到另一个 SM（或排队）
  ├── ...
  └── Block N-1
```

### 3.2 代码怎么写

```cuda
// 这是一个 CUDA kernel（在 GPU 上运行的函数）
__global__ void add_vectors(const float* a, const float* b, float* c, int n) {
    // 计算当前线程的全局 ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 调用（在 CPU 上写）
int main() {
    const int N = 1024 * 1024;
    float *d_a, *d_b, *d_c;

    // 1. 在 GPU 上分配内存
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 2. 把数据从 CPU 拷贝到 GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 启动 kernel
    dim3 block(256);                    // 每个 block 256 个线程
    dim3 grid((N + 255) / 256);         // 需要多少个 block 覆盖所有元素
    add_vectors<<<grid, block>>>(d_a, d_b, d_c, N);

    // 4. 等 GPU 做完
    cudaDeviceSynchronize();

    // 5. 把结果拷回 CPU
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
}
```

### 3.3 关键内置变量

| 变量 | 含义 | 例子 |
|------|------|------|
| `threadIdx.x` | 线程在 block 内的编号 | 0, 1, ..., blockDim.x-1 |
| `blockIdx.x` | block 在 grid 内的编号 | 0, 1, ..., gridDim.x-1 |
| `blockDim.x` | block 包含多少线程 | 你指定的值（如 256） |
| `gridDim.x` | grid 包含多少 block | 你指定的值 |
| `warpSize` | 一个 warp 的线程数 | 32（永远不变） |

换算公式：

```
全局线程 ID = blockIdx.x * blockDim.x + threadIdx.x
总线程数    = gridDim.x * blockDim.x
```

### 3.4 二维和三维

```cuda
// 二维 grid，适合处理图片
dim3 block(16, 16);                     // 16×16 = 256 线程
dim3 grid((W+15)/16, (H+15)/16);         // 覆盖整张图片

__global__ void process_image(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;             // 展平为线性地址
    // ...
}
```

---

## 第四章：动手写第一个 CUDA 程序

### 4.1 环境检查

```bash
nvcc --version          # 检查 CUDA 编译器
nvidia-smi              # 检查 GPU 状态
```

### 4.2 编译和运行

```bash
# 编译
nvcc -arch=sm_110 -o my_program my_program.cu

# 运行
./my_program
```

**nvcc 编译流程：**
1. 把 `.cu` 文件中的 GPU 代码（`__global__` 函数）编译为 PTX（中间汇编）
2. 把 PTX 编译为 SASS（GPU 实际执行的二进制）
3. 把 CPU 代码（`main` 函数）用 g++ 编译
4. 链接在一起

### 4.3 常用 NVCC 选项

```
-arch=sm_110      生成 sm_110（Thor）的 SASS
-O3               最大优化
-std=c++17        C++ 标准
-lineinfo         调试信息（配合 ncu 用）
--ptxas-options=-v  显示寄存器/共享内存使用情况
```

---

## 第五章：内存模型

### 5.1 层次结构

```
速度最快    容量最小
Register    ← 每个线程私有，~255 个 32-bit 寄存器
  ↑
Shared Mem  ← 同一个 block 内线程共享，几十 KB
  ↑
L1 Cache    ← 每个 SM 私有
  ↑
L2 Cache    ← 所有 SM 共享，几 MB
  ↑
Global Mem  ← 显存，几十 GB，延迟 ~200-800 cycles
速度最慢    容量最大
```

### 5.2 Global Memory（全局内存）

```cuda
float *d_ptr;
cudaMalloc(&d_ptr, size);     // 分配
cudaMemcpy(d_ptr, src, size, cudaMemcpyHostToDevice); // 拷入
cudaMemcpy(dst, d_ptr, size, cudaMemcpyDeviceToHost); // 拷出
cudaFree(d_ptr);              // 释放
```

**特点：**
- 所有线程都能访问
- 容量大（几十 GB）
- 速度慢（~200-800 cycles 延迟）
- 没有缓存一致性保证

### 5.3 Shared Memory（共享内存）

```cuda
__global__ void kernel() {
    __shared__ float sdata[256];  // 每个 block 独立的一份

    int idx = threadIdx.x;
    sdata[idx] = some_value;
    __syncthreads();              // 等所有线程写完

    // 现在可以读其他线程写的数据
    float val = sdata[(idx + 1) % 256];
}
```

**特点：**
- 同一个 block 内的线程共享
- 速度非常快（~1-2 cycles 延迟）
- 容量极小（通常 48-164 KB per SM）
- 需要 `__syncthreads()` 同步

**Shared Memory 是优化的核心工具**：把数据从 global 搬到 shared，反复使用，避免反复读 global。

### 5.4 寄存器

```cuda
__global__ void kernel() {
    float x = 1.0f;       // x 存在寄存器里
    float y = x + 2.0f;   // y 也在寄存器里
    // ...
}
```

**特点：**
- 最快（0 cycle 延迟）
- 每个线程私有
- 总量有限（每个 SM 固定，线程太多时→寄存器溢出到 local memory→变慢）

---

## 第六章：性能基础

### 6.1 Occupancy（占用率）

**定义：** 活跃 warp 数 ÷ SM 最大 warp 数

```
Thor 每个 SM 最多 32 个 warp（= 1024 线程）

如果你每个 block 用 256 线程：
  每个 SM 能同时跑 4 个 block
  4 × 8 = 32 个 warp → 100% occupancy

如果你每个 block 用 32 线程：
  每个 SM 能同时跑 32 个 block（理论上限）
  32 × 1 = 32 个 warp → 100% occupancy
  
但如果每个线程用很多寄存器或 shared memory：
  → 能同时跑的 block 减少 → occupancy 降低
```

**为什么 occupancy 重要？**

因为 warp 在等数据时（访存延迟），scheduler 切换到另一个 warp。warp 越多，延迟越容易被隐藏。

```
100% occupancy: 32 个 warp，一个等数据，切到另一个
  → 延迟被完美隐藏

12% occupancy: 4 个 warp，都在等数据时 → SM 空转
  → 性能暴跌
```

### 6.2 Memory Coalescing（合并访问）

**这是最容易被忽视的性能杀手。**

```
warp 内 32 个线程访问 global memory 时：

合并访问（连续地址）：
  Thread 0 → addr 0      ┐
  Thread 1 → addr 4      │  硬件合并成
  Thread 2 → addr 8      ├─ 一次 128 字节传输
  ...                     │
  Thread 31 → addr 124   ┘

非合并访问（跳跃地址）：
  Thread 0 → addr 0      ┐
  Thread 1 → addr 128    │  拆成 32 次
  Thread 2 → addr 256    ├─ 单独的 4 字节传输
  ...                     │
  Thread 31 → addr 3968  ┘  带宽利用率暴跌
```

**规律：** warp 内 thread i 访问 `base + i * stride`，stride=1 时最快。

### 6.3 Arithmetic Intensity（算术强度）

```
AI = FLOPs ÷ Bytes

高 AI（> 50）：  计算密集，瓶颈在计算能力
低 AI（< 10）：  访存密集，瓶颈在带宽
```

**Roofline 模型：**

```
Performance ↑
            │  ┌────────────────── 算力天花板
            │  │                  /
            │  │                 /  ← 实际 kernel 的位置
            │  │                /
            │  ┌───────────────/─── 带宽天花板
            │  │              /
            │  │             /
            └─┴────────────/──────→ AI (FLOP/Byte)
               ridge point
```

- 在 ridge point 左边 → **memory-bound**（等数据）
- 在 ridge point 右边 → **compute-bound**（等计算）

### 6.4 优化策略总结

| 问题 | 诊断方法 | 解决方案 |
|------|---------|---------|
| Low occupancy | ncu 查看 Occupancy | 增大 block size / 减少每个线程的寄存器使用 |
| Non-coalesced | ncu 查看 L1/TEX 吞吐 | 调整数据布局（AoS→SoA） |
| Memory-bound | ncu 查看 Mem Pipes Busy | Tiling / Fusing / 提高 AI |
| Compute-bound | ncu 查看 SM Busy | 降低精度 / Tensor Core |
| 启动开销大 | kernel 耗时 < 10 µs | CUDA Graph / 合并 kernel |

---

## 第七章：常用工具

### 7.1 CUDA Event（计时）

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start, 0);
cudaEventCreate(&stop, 0);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**不要用 clock() 或 gettimeofday() 测 GPU kernel**，它们只测 CPU 端时间。

### 7.2 Nsight Systems（nsys）

```bash
# 获取 GPU kernel 时间线
nsys profile --trace=cuda ./my_program

# 查看每个 kernel 的耗时统计
nsys stats --report cuda_gpu_kern_sum report.nsys-rep

# 查看 kernel 时间线
nsys stats --report cuda_gpu_trace report.nsys-rep
```

nsys 告诉你：**时间花在哪个 kernel 上**（宏观）。

### 7.3 Nsight Compute（ncu）

```bash
# 获取硬件计数器
ncu --set basic ./my_program

# 关键指标：
#   Occupancy          → 占用率
#   SM Busy            → SM 忙碌程度（计算瓶颈？）
#   Mem Pipes Busy     → 访存单元忙碌程度（访存瓶颈？）
#   L1/TEX Hit Rate    → 缓存命中率
#   Dram Throughput    → 显存带宽利用率
```

ncu 告诉你：**kernel 为什么慢**（微观）。

---

## 第八章：常见 CUDA 模式

### 8.1 Reduction（规约）

把 N 个数加在一起：

```cuda
__global__ void reduce(const float* in, float* out, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. 加载到 shared memory
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // 2. 树形规约
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 写结果
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
```

关键点：
- Shared memory 让数据在 block 内共享
- 树形规约 O(log N) 而不是 O(N)
- `__syncthreads()` 确保所有线程写完了再读

### 8.2 Tiled Matmul（分块矩阵乘）

```cuda
__global__ void tiled_matmul(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < K / TILE; t++) {
        // 协作加载 tile 到 shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        // 从 shared memory 计算
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

为什么分块快？因为每块数据从 global 加载一次，然后在 shared memory 中被复用 TILE 次。

### 8.3 Tensor Core 简介

NVIDIA 从 Volta 架构开始加入了 Tensor Core——专门做矩阵乘的硬件单元。

```cuda
// 你要写的是 WMMA API
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_matmul(half* A, half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 加载数据
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, N);

    // 一条 mma 指令 = 8192 次浮点运算
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 写回
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}
```

**一条 mma_sync = 16×16×16 的矩阵乘 = 8192 FLOPs = 一条指令。**

对比 FP32 的一条 FMA：
- FMA：2 FLOPs/指令
- Tensor Core mma_sync：8192 FLOPs/指令

**但 Tensor Core 只能算 16×16 的块，不能直接算任意大小。**

---

## 第九章：完整的 GPU 知识体系

### 9.1 硬件架构演进

| 架构 | Compute Capability | 特点 |
|------|-------------------|------|
| Volta | sm_70 | 引入 Tensor Core（第一代） |
| Turing | sm_75 | 引入 INT8 Tensor Core |
| Ampere | sm_80/sm_86 | 第三代 TC，支持 TF32/BF16 |
| Hopper | sm_90 | 第四代 TC，Transformer Engine |
| Blackwell | sm_100 | 第五代 TC，FP4/FP6 |
| Thor | sm_110 | 车载版 Blackwell |

### 9.2 性能计算速查

```cuda
// 计算 throughput
float ms = elapsed_ms;
double flops = 2.0 * M * N * K;          // matmul FLOPs
double gflops = flops / ms / 1e6;         // GFLOP/s

// 计算带宽
double bytes = n * sizeof(float) * 2;     // read + write
double gbps = bytes / ms / 1e6;           // GB/s

// 计算 AI
double ai = flops / bytes;                // FLOP/Byte

// 从 roofline 推断瓶颈
// 如果 gflops ≈ bw_peak × ai → memory-bound
// 如果 gflops ≈ compute_peak → compute-bound
```

### 9.3 常见陷阱

1. **忘了 cudaDeviceSynchronize()** → 计时不准
2. **kernel 里有分支导致 warp divergence** → 性能骤降
3. **stride 访问** → 带宽利用率低
4. **block 线程数太少** → occupancy 低
5. **寄存器溢出** → 编译器把寄存器存到 local memory（等于 global memory 速度）
6. **shared memory 超额** → kernel 启动失败（或者 occupancy 暴跌）
7. **没检查 cudaError** → 出错时静默失败

### 9.4 学习路径

```
1. 本教程 → 理解基本概念
2. nano-npu-profile/00_tutorial.py → 用 Triton 感受 GPU 编程
3. CUDA C++ 官方 samples → vectorAdd、reduction、matmul
4. nano-npu-profile/01~05 → 用实测理解性能
5. 自己写一个 mini 项目（如：自己实现一个简单的卷积层）
```

---

## 第十章：术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| SM | Streaming Multiprocessor | GPU 的计算单元，包含 CUDA Core、Shared Memory、Warp Scheduler |
| CUDA Core | — | SM 中的 ALU，执行算术指令 |
| Warp | — | 32 个线程一组，硬件执行的基本单位 |
| Block | Thread Block | 一组线程，在同一个 SM 上执行，可共享内存 |
| Grid | — | 一组 Block，覆盖整个问题 |
| Occupancy | 占用率 | 活跃 warp ÷ 最大 warp，越高延迟隐藏越好 |
| Coalescing | 合并访问 | warp 内线程访问连续内存，硬件合并为一次传输 |
| Divergence | 分支发散 | warp 内线程走不同分支，串行执行 |
| Shared Memory | 共享内存 | 片上快速内存，block 内共享 |
| Global Memory | 全局内存 | 显存，容量大速度慢 |
| Register Spill | 寄存器溢出 | 寄存器不够用，数据被存到 local memory（慢） |
| AI | Arithmetic Intensity | FLOPs/Byte，衡量计算密集度 |
| Roofline | 屋顶线模型 | 带宽和算力的二维性能分析模型 |
| Tensor Core | 张量核心 | 专用矩阵乘单元，一条指令 8192 FLOPs |
| WMMA | Warp Matrix Multiply-Accumulate | Tensor Core 的 CUDA API |
| PTX | Parallel Thread eXecution | CUDA 中间汇编（类似 Java bytecode） |
| SASS | Streaming ASSembly | GPU 实际执行的二进制指令 |
| Launch Overhead | 启动开销 | CPU 到 GPU 的 kernel 启动时间 (~5-10 µs) |
| CUDA Graph | — | 预编译 kernel 执行图，消除启动开销 |

---

## 写在最后

**GPU 编程的核心心法：**

1. **数据搬运比计算贵。** 尽量把数据留在 GPU 上，减少 CPU-GPU 传输。
2. **每次读 global memory 都要想：这次读的值用了几次？** 只用一次就是浪费。
3. **Warp 是上帝。** 所有优化归根结底是让 warp 高效工作。
4. **Profile 之后再优化。** 靠猜不如靠 ncu/nsys。
5. **先让代码正确，再让它快。** 不正确的快没有意义。
