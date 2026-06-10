# 01: CUDA Graph — 消除 Kernel 启动开销

## 0. 这条线是怎么连起来的

```
你在前面亲手测出了两个问题：

Step 04 mini_cnn 的 nsys 时间线：
  kernel1: conv1      ████████████████  28 µs
  ───────────── gap ──║                  ~5 µs (启动开销)
  kernel2: relu1      ██                 3 µs
  ───────────── gap ──║                  ~5 µs
  kernel3: pool1      ███                5 µs
  ───────────── gap ──║                  ~5 µs
  kernel4: conv2      ████████████████  28 µs
  ...

  6 个 kernel × ~5 µs = 30 µs 纯浪费
  batch=1 时占总时间 30-50%

  → 你在 Step 04 看到了这个，但不知道它叫什么、怎么修

Step 05 的瓶颈分类：
  memory-bound   ✓ ✓ ✓
  compute-bound  ✓ ✓ ✓
  launch-bound   ? ←── 我们漏了这类！

  → 你学了 compute/memory/occupancy，但没学 "启动开销瓶颈"

alexnet_profile.py 的 nsys 结果：
  cuBLAS GEMV (FC 层) = 92.4% GPU 时间
  每个 FC 层都是 tiny kernel（1-10 µs）
  启动开销占比极高

  → 这就是为什么 FC 层是最大瓶颈——不是因为算得慢，是因为启动太多次

06_Triton/04_mini_cnn.py:
  用 Python 写的 CNN，440 inf/s（比 C++ 版慢 8x）
  原因之一：Python 本身的解释器开销 + 每次调 Triton kernel 的启动开销

  → 如果不用 Graph，Python 启动 kernel 比 C++ 更慢

这三个问题都指向同一个解法：CUDA Graph
```

## 1. 启动开销到底多大？

回到 Step 01 的 `bw_bench`：为什么小数据量的带宽那么低？

```
数据量      带宽      解释
1 KB        2 GB/s   →  启动开销占了 99% 时间
1 MB       50 GB/s   →  启动开销还是占很大
64 MB     219 GB/s   →  终于饱和了
```

**同一个原理**：kernel 越小，启动开销占比越大。

在 mini_cnn 里：

```
一个空 kernel（什么都不做）：         ~3-5 µs
mini_cnn 的 relu kernel (3 µs)：      启动 5 µs + 计算 3 µs = 62% 浪费
mini_cnn 的 conv kernel (28 µs)：     启动 5 µs + 计算 28 µs = 15% 浪费
```

**推理时 batch=1，模型越小，启动开销越致命。**

### 在 roofline 上怎么看？

kernel 耗时极短（<10 µs）且在 roofline 上远低于任何天花板 → **launch-bound**

```
GFLOP/s ↑
        │
  天花板 ┼───────────────────
        │
  低    ┼──●──── 不是因为算力不够或带宽不够
        │        是因为启动开销稀释了有效时间
        └────────────────────→ AI
```

这和 Step 02 的 roofline 不同 —— roofline 只告诉你 compute-bound 还是 memory-bound，**没告诉你 launch-bound**。这是个 roofline 模型也不覆盖的盲区。

## 2. CUDA Graph 原理

### 核心思想

```
传统方式（每次启动都走完整路径）：
  CPU:  写参数1 → 通知 GPU → 写参数2 → 通知 GPU → ...
  GPU:            接收 → 解析 → 执行 → 接收 → 解析 → 执行 → ...
        ↑ CPU 和 GPU 来回握手，每次 ~3-5 µs

CUDA Graph（一次构建，无限重放）：
  Step 1（capture）：把 kernel 启动"录下来"（参数、顺序、依赖）
  Step 2（replay）：  直接执行录好的内容，没有 CPU 参与
                      kernel 之间零延迟
```

注意这和 Step 04 里学的 **pipeline buffer** 没有关系。Pipeline 解决的是"算得快不快"，Graph 解决的是"启动快不快"。

### 为什么快？

回到 Step 04 LECTURE.md 里的工厂类比：

```
传统方式：每个工人干完活，去找厂长要下一个任务
          → 每步都有沟通开销

CUDA Graph：厂长提前写好一整天的任务清单
            工人直接看清单干活，不用问
             → 沟通开销 = 0
```

## 3. CUDA C++ API

### 基本用法

```cuda
cudaGraph_t graph;
cudaGraphExec_t instance;

// Step 1: 开始录制
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Step 2: 正常写 kernel 启动（会被录下来）
for (int i = 0; i < N; i++) {
    kernel<<<grid, block, 0, stream>>>(args...);
}

// Step 3: 停止录制，拿到 graph
cudaStreamEndCapture(stream, &graph);

// Step 4: 实例化（编译成 GPU 可执行格式）
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// Step 5: 重放 —— 没有启动开销
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);   // 1 次启动触发整个图
}
```

### 完整 demo 在 `demo_graph.cu`

这个 demo 对比了传统方式 vs CUDA Graph 启动 1000 次 kernel。**它直接模拟了 mini_cnn 的问题**：很多小 kernel 串行启动时的累积开销。

## 4. PyTorch 中的 CUDA Graph

### 手动版本

```python
import torch

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = model(x)          # 录制整个 forward

# 后续推理
for _ in range(1000):
    graph.replay()        # 零开销重放
```

### torch.compile 自动版本

```python
# PyTorch 2.0+
model = torch.compile(model, mode="reduce-overhead")
y = model(x)              # 第一次运行 = 编译 + 图捕获
y = model(x)              # 后续 = 图重放
```

`mode="reduce-overhead"` 背后就是 CUDA Graph。**它把你 Step 04 mini_cnn 的 6 个 kernel 录制成一个图，然后无限重放。**

## 5. 和前面步骤的具体连接

### 连接 Step 04 mini_cnn

```
mini_cnn 当前（C++）:
  6 次 cudaLaunchKernel (6 × ~5 µs = 30 µs)
  3700 inf/s

mini_cnn + CUDA Graph:
  1 次 cudaGraphLaunch + 6 个 kernel 连续执行
  预计: ~4500 inf/s (+20%)

  为什么只提升 20%？
  → 因为 FC 层占 84.6% 的时间
  → FC 层本身是 compute-bound（181 µs）
  → 启动开销只影响剩下 15.4% 的时间
```

### 连接 AlexNet case

```
alexnet_profile.py 的 nsys 数据：
  cuBLAS GEMV (FC 层) = 92.4% GPU 时间
  但这些 GEMV 每个都很小（M=1, K=4096, N=1000）
  每次调用 cuBLAS 都有 ~5 µs 启动开销

  → 传统 PyTorch 里每个 FC 层 = 1 次 cuBLAS call
  → torch.compile + CUDA Graph = 全部 FC 层合成 1 次图执行

  alexnet 实测加速预计: +15-30% (batch=1)
```

### 连接 06_Triton

```
Triton 的 Python kernel 启动开销比 C++ 更大：
  C++:   kernel<<<>>>         ~3-5 µs
  Python: triton.jit func()   ~5-10 µs (Python 解释器 + 启动)

  Triton 版 mini_cnn 只有 440 inf/s
  C++ 版 mini_cnn 有 3700 inf/s
  差距原因之一：Triton 每次调用有更大的 Python 层开销

  Triton + CUDA Graph（通过 torch.compile）:
    → 消除 Python 层开销
    → 消除 kernel 间启动开销
    → 预计: 从 440 提升到 2000+ inf/s
```

### 连接 Step 05 瓶颈分类

Step 05 把瓶颈分成 memory-bound, compute-bound, occupancy。**缺少一类：**

```
Step 05 的分类:
  │
  ├── memory-bound     (BW 不够)
  ├── compute-bound    (算力不够)
  └── low occupancy    (SM 没喂饱)

漏了的：
  launch-bound        (启动开销稀释了有效计算时间)
                      症状：kernel 时间 < 10 µs
                      nsys 上看到明显的 kernel 间 gaps
                      解决：CUDA Graph / kernel fusion
```

## 6. 什么时候应该/不该用

| 场景 | kernel 典型大小 | 启动开销占比 | 用 Graph？ |
|------|----------------|-------------|-----------|
| mini_cnn, batch=1 | 3-181 µs (平均 ~40) | ~20-40% | **推荐** |
| AlexNet FC 层 | ~10 µs | ~30-50% | **推荐** |
| AlexNet Conv 层 | ~200 µs | ~5% | 可忽略 |
| ResNet-50, batch=32 | 1-50 ms | <1% | 不需要 |
| 训练 (大 batch) | 1-100 ms | <1% | 不需要 |

## 7. 总结

```
你在 Step 01 测了 BW，Step 02 画了 roofline，Step 03 优化了 matmul，
Step 04 看了 mini_cnn 的 nsys trace，Step 05 学了瓶颈分类。

nsys trace 上 kernel 之间的 gaps —— 你现在知道那叫 launch overhead。

CUDA Graph = 消除这些 gaps 的工具。
torch.compile + reduce-overhead = 一行代码搞定。

下一章：torch.compile —— 不仅消除 gaps，还能融合 kernel。
```
