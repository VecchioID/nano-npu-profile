# Step 04: 端到端模型 Profiling

## 前置知识：GPU 基础架构

### 三层结构：Thread → Warp → Block → SM

```
CUDA 程序启动 kernel：
  kernel<<<grid, block>>>()
    grid  = 几个 block（由你指定）
    block = 几个 warp（由你指定，通常 128-256 线程）
    warp  = 32 个线程（硬件固定，不可更改）

GPU 硬件执行：
  SM（Streaming Multiprocessor）← 负责执行 block
    ├── 每个 SM 有多个 CUDA Core（计算单元）
    ├── 每个 SM 可以同时跑多个 block（取决于资源）
    ├── block 里的线程→分成 warp→在 CUDA Core 上执行
    └── SM 之间的 block 是**独立并行**的
```

### 关键概念

| 术语 | 解释 |
|------|------|
| **Thread（线程）** | 最小的执行单元，每个线程跑同样的代码但处理不同的数据（SIMT） |
| **Warp（线程束）** | 32 个线程一组，**硬件的真正执行单位**。一个 warp 里所有线程同时执行同一条指令。如果线程走不同分支（if/else），就**串行化**（warp divergence） |
| **Block（线程块）** | 一组线程（你指定的数量），block 内的线程可以通过 shared memory 通信、通过 `__syncthreads()` 同步 |
| **Grid** | 一组 block，覆盖整个问题规模 |
| **SM** | GPU 的物理计算单元。一个 SM 包含多个 CUDA Core、shared memory、寄存器、warp scheduler 等。一个 block **只能在一个 SM 上**跑，但一个 SM 可以同时跑**多个 block** |

### 软件 vs 硬件：完整映射图

```
你写的 CUDA 代码（软件概念）         GPU 执行的（硬件实体）
─────────────────────────────      ─────────────────────────
                                   
Grid（你指定几个 block）             
  │                                 
  ├─ Block 0                          → SM 0（一个 SM 执行一个 block）
  │   ├─ Thread 0                         → CUDA Core 0
  │   ├─ Thread 1                         → CUDA Core 1
  │   ├─ ...                              → ...
  │   ├─ Thread 31                  ← Warp 0（32 个线程一组，在同一时刻执行同一条指令）
  │   ├─ Thread 32                        → CUDA Core 0（下一个 cycle）
  │   ├─ ...                              → ...
  │   ├─ Thread 63                  ← Warp 1
  │   ├─ ...（直到 Thread 255 = 8 个 warp）
  │
  ├─ Block 1                          → SM 1（如果还有空闲 SM）
  ├─ Block 2                          → SM 2
  ├─ ...
  └─ Block 19                         → SM 19（Thor 有 20 个 SM）
                                       └── 如果 grid 有 40 个 block：
                                           SM 0 先跑 Block 0，跑完再跑 Block 20
                                           SM 1 先跑 Block 1，跑完再跑 Block 21
                                           ...以此类推
```

### 关键规则：一个 block 只能在一个 SM 上

**Block 是分配给 SM 的，不是切分的。** 一个 block 的所有线程都在同一个 SM 上执行。SM 有多个，所以多个 block 可以并行。

### 用 Grid 和 Block 的实际例子

**例 1：thor, 20 SM, 处理 1000 个元素的向量加法**

```cuda
dim3 block(256);       // 256 线程 = 8 个 warp
dim3 grid(4);          // 4 个 block（4 × 256 ≈ 1000 个元素）

kernel<<<grid, block>>>(...);
```

执行过程：
```
Step 1: SM 0 拿到 Block 0，SM 1 拿到 Block 1，SM 2 拿到 Block 2，SM 3 拿到 Block 3
        → 4 个 SM 同时工作，16 个 SM 空闲
Step 2: 所有 block 执行完毕，kernel 结束
```

**例 2：同样 1000 个元素，但启用了足够大的 grid**

```cuda
dim3 block(256);
dim3 grid(20);         // 20 个 block

kernel<<<grid, block>>>(...);
```

```
Step 1: 20 个 SM 各拿 1 个 block → 全部 SM 同时工作！
```

**例 3：FC 层的问题（Step 04 的核心）**

```cuda
dim3 block(256);
dim3 grid(1);          // 只有 1 个 block！

kernel<<<grid, block>>>(...);
```

```
Step 1: SM 0 拿到 Block 0 → 1 个 SM 工作，19 个空闲
        而且 Block 0 里 256 个线程，只有 10 个有用
        → 利用率 = 1/20 SM × 10/256 线程 = 0.2%！
```

### 类比：工厂流水线

把 GPU 想象成一个有 20 条产线的工厂：

| GPU | 工厂 | FC 层的问题 |
|-----|------|-------------|
| SM | 一条产线 | 你只开了 1 条产线 |
| Block | 一个订单 | 你只下了 1 个订单 |
| Thread | 一个工人 | 你招了 256 个工人，但只让 10 个干活 |
| Warp | 一组配合的工人（必须同时做同一件事） | 其余 246 个工人在旁边看着 |

**正确的做法：把订单拆成多个小单（多个 block），让所有产线（SM）同时跑。**

对于 FC 层：输出 10 个神经元 → 应该拆成 10 个 block（每个 block 算一个神经元），而不是 1 个 block 算所有。

### 为什么 Tensor Core 测出 50 TFLOPS 但这里只有 12 GFLOP/s？

```
纯 TC benchmark:  80 block × 100K 次 mma_sync  →  20 SM 全跑满
mini_cnn FC 层:    1 block × 10 个有用线程     →  1 个 SM 只用了 3% 的线程

相差：20 SM × 33× 更少线程 = 660 倍利用率的差距
```

**GPU 快是因为并行。如果只用 1 个 block，GPU 就变成了一个慢 CPU。**

### 补充：Block 和 SM 的资源限制

一个 SM 能同时跑多少个 block 取决于：
1. **线程数上限**（Thor 每个 SM 最大 1024 线程 = 4 block × 256）
2. **shared memory 上限**（每个 SM 的 shared memory 是固定的，如果每个 block 用很多 shared memory，能同时跑的 block 就少）
3. **寄存器上限**（每个 SM 寄存器总数固定，如果每个线程用很多寄存器，能同时跑的线程就少）

这就是 **occupancy（占用率）** 的概念：实际活跃 warp 数 ÷ SM 最大 warp 数。

---

## 概述

前面我们测的都是单个算子。现在把它们拼成**一个真正的 CNN**，看看模型端到端的表现。

```
输入 (32×32×3 RGB)
  → Conv1 (3×3, 3→16)
  → ReLU
  → MaxPool (2×2, 32×32→16×16)
  → Conv2 (3×3, 16→32)
  → ReLU
  → FC (8192→10)
  → 输出 (10 类)
```

这是一个极小的分类 CNN。故意做小是因为即使 kernel 写得很 naive 也能跑快——你可以快速迭代、马上看到效果。

---

## 第一部分：基线

### 编译和运行

```bash
cd 04_model_level
nvcc -arch=sm_110 -std=c++17 -O3 -o mini_cnn mini_cnn.cu
./mini_cnn
```

输出：
```
Total inferences: 100
Total time: 27.31 ms
Avg time per inference: 0.273 ms
Throughput: 3661.3 inferences/sec
Estimated GFLOP/s: 12.63
```

**每秒 3,661 次推理，12.6 GFLOP/s。**

对比理论天花板：FP32 天花板约 6,500 GFLOP/s（来自 Step 01）。我们只用了 GPU 计算能力的 **0.2%**。为什么？

### 各层耗时分布（nsys）

我们用 `nsys` 做了 profile，拿到每个 kernel 的耗时：

| Kernel | 时间占比 | 总耗时 (ms) | 调用次数 | 平均 (µs) |
|--------|----------|-------------|----------|-----------|
| fc_layer | 84.6% | 20.43 | 100 | 204.3 |
| conv_layer | 13.4% | 3.23 | 200 | 16.1 |
| relu_layer | 1.1% | 0.25 | 200 | 1.3 |
| maxpool_layer | 1.0% | 0.25 | 100 | 2.5 |

**FC 层吃掉了 84.6% 的 GPU 时间。** 这就是瓶颈。

---

## 第二部分：瓶颈分析

### FC 层的问题

FC kernel：
```cuda
__global__ void fc_layer(const float* input, const float* weight,
                         const float* bias, float* output,
                         int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim) return;
    float sum = bias[idx];
    for (int i = 0; i < in_dim; i++)
        sum += input[i] * weight[idx * in_dim + i];
    output[idx] = sum;
}
```

启动参数：
```cuda
dim3 block_f(256);              // 256 线程/block
dim3 grid_f((10 + 255) / 256);  // 1 个 block（10 个输出神经元，ceil(10/256) = 1）
```

问题一目了然：

```
out_dim = 10, threads = 256, blocks = 1

只有 10 个线程在做有用功。246 个线程 → 直接返回。
只有 1 个 block → 只有 1 个 SM 在工作 → 19 个 SM 空闲。
```

每次 FC 推理做 `2 × 10 × 8192 = 163,840 FLOPs`。204 µs 一次，算下来只有 `163,840 / 204e-6 = 0.80 GFLOP/s` —— **远低于任何天花板**。

### 启动开销问题

就算忽略计算本身：一个 CUDA kernel 启动就有 ~5-10 µs 的开销（CPU 到 GPU 的握手时间）。我们的 FC kernel 最快 181 µs —— 所以**真正的计算**只占其中极小一部分。

## 第三部分：模型的 Arithmetic Intensity

各层的算术强度：

| 层 | FLOPs | 读取字节数 | AI (FLOP/Byte) |
|-------|-------|------------|-----------------|
| Conv1 (3×3, C=3→K1=16) | 884,736 | H×W×C×4 + K1×C×9×4 = 12,288+1,728 = 14,016 | 63.1 |
| ReLU1 | 16,384 | 16,384×4 = 65,536 | 0.25 |
| MaxPool1 | 16,384 | 16,384×4 = 65,536 | 0.25 |
| Conv2 (3×3, K1=16→K2=32) | 2,359,296 | 65,536+18,432 = 83,968 | 28.1 |
| ReLU2 | 8,192 | 8,192×4 = 32,768 | 0.25 |
| FC (8192→10) | 163,840 | 32,768+327,680 = 360,448 | 0.45 |

关键观察：Conv 层有**高 AI**（28-63 FLOP/Byte），但 FC 的 AI 只有 **0.45**。整个模型的 AI 被 FC 层拖垮。

### Roofline 定位

在我们的 roofline 上：
- Conv 层（AI ~28-63）→ 靠近带宽天花板区域，优化好能达到 ~600-1400 GFLOP/s
- FC 层（AI ~0.45）→ 深度 memory-bound，理论上限 ~99 GFLOP/s（219 × 0.45）
- 但我们实测只有 0.80 GFLOP/s —— 连带宽天花板都没达到

所以 FC 的瓶颈不只是 memory-bound —— 是**利用率不足**。连带宽都没用满，因为只有 10 个线程在工作。

---

## 第四部分：Profiling 方法论

### nsys 告诉你什么

| 命令 | 获取内容 |
|---------|-------------|
| `nsys profile --trace=cuda ./mini_cnn` | GPU 时间线、kernel 耗时、API 调用 |
| `nsys stats --report cuda_gpu_trace` | 每个 kernel 的开始/结束/时长（纳秒级） |
| `nsys stats --report cuda_gpu_kern_sum` | 汇总统计：总时间、调用次数、平均/中位/最小/最大值 |

### ncu（如果可以访问的话）

ncu 给出的硬件计数器：占用率、SM 吞吐量、显存吞吐量、缓存命中率。但在这个系统上需要 root 权限（ERR_NVGPUCTRPERM）。

### 黄金法则

**先在 profile 里找到瓶颈，再动手优化。** 靠猜是不靠谱的——最慢的层往往不是你预期的那一个。

---

## 第五部分：优化策略

### 1. 修复 FC Kernel（最大收益）

**问题**：1 block × 256 线程，只有 10 个线程干活。

**方案**：把归约逻辑放进 kernel —— 每个线程算部分和，然后 warp 内规约：

```cuda
__global__ void fc_fast(const float* input, const float* weight,
                        const float* bias, float* output,
                        int in_dim, int out_dim) {
    int tid = threadIdx.x;
    int out_id = blockIdx.x;
    if (out_id >= out_dim) return;

    float sum = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x)
        sum += input[i] * weight[out_id * in_dim + i];
    // warp 内归约
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (tid == 0) output[out_id] = sum + bias[out_id];
}
```

这样所有 256 个线程都参与计算，并且启动 10 个 block（每个 SM 一个，没有空闲 SM）。

### 2. 把 ReLU 融合进 Conv

ReLU 就是 `data[i] = max(0, data[i])` —— 一个极简单的逐元素操作。每次调用耗时 ~1.3 µs，主要是 kernel 启动开销。把它融合到 conv kernel 里（累加后直接 apply）→ 省掉 100 次 kernel 启动。

### 3. 使用 Tensor Core

Conv1: 3×3 kernel, C=3 → Tensor Core 要求 C >= 16（或者特殊处理）。Conv2: C=16 → 可以用 WMMA。但矩阵太小，启动开销会占主导。

### 4. 增大模型

这个模型故意做得很小以方便快速迭代。真实部署中：
- 输入: 224×224×3（像素多 49 倍）
- 通道数: 64+（多 4 倍）
- 结果：矩阵更大，GPU 利用率更高

---

## 总结

| 概念 | 核心要点 |
|---------|-------------|
| 端到端 profiling | 测量所有层，找到瓶颈 |
| Kernel 启动开销 | 每次 ~5-10 µs，6 个 kernel × 100 次迭代累积可观 |
| 利用率不足 | 1 block × 10 活跃线程 = 0.2% SM 利用率 |
| FC 层陷阱 | 输出维度小 + 输入维度大 = naive kernel 效率极低 |
| Profiling 方法论 | nsys 看时间线，ncu 看硬件计数器（如果可以） |
| 优化优先级 | 先修最慢的层（FC），再融合小 kernel |

**Step 04 最核心的一课：一个模型的快慢取决于它最慢的那一层。Profile 找到它，然后修复它。**

---

## 附录：代码详解

### `mini_cnn.cu` 结构

```
main()
  ├── 分配设备内存
  ├── 循环 100 次：
  │   ├── conv_layer (Conv1: 32×32×3 → 32×32×16)
  │   ├── relu_layer (逐元素 max(0,x))
  │   ├── maxpool_layer (2×2, 32×32 → 16×16)
  │   ├── conv_layer (Conv2: 16×16×16 → 16×16×32)
  │   ├── relu_layer (逐元素 max(0,x))
  │   └── fc_layer (8192 → 10)
  ├── 同步、计时
  └── 打印结果
```

### 内存布局

```
d_input:   32×32×3 floats  = 12,288 bytes
d_w1:      16×3×3×3 floats = 1,728 bytes
d_b1, d_b2: K 个 float
d_l1_out:  16×32×32 floats = 65,536 bytes
d_pool_out: 16×16×16 floats = 16,384 bytes
d_l2_out:  32×16×16 floats = 32,768 bytes
d_w3:      10×8192 floats  = 327,680 bytes
```

总权重：~340 KB（轻松放进 L2 cache）。这个模型是 **weight-bound**（权重小），不是 bandwidth-bound。

### nsys 时间线（单次迭代示例）

```
355,253,697 ns  conv_layer   (64 blocks × 16×16, 5.6 µs)
355,261,921 ns  relu_layer   (1 block × 256, 1.2 µs)
355,266,017 ns  maxpool_layer (16 blocks × 16×16, 2.5 µs)
355,270,145 ns  conv_layer   (32 blocks × 16×16, 26.5 µs)
355,298,785 ns  relu_layer   (1 block × 256, 1.2 µs)
355,302,945 ns  fc_layer     (1 block × 256, 181 µs) ← 比 conv 慢 32 倍！
```

`conv_layer`（Conv1 中 5.6 µs，Conv2 中 26.5 µs）与 `fc_layer`（181 µs）之间的差距说明 FC 是一个 **32 倍的尖峰** —— 瓶颈清晰可见。
