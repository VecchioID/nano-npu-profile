# 05: CUDA Streams — 让 GPU 同时做多件事

## 0. 这条线是怎么连起来的

```
Step 04 mini_cnn 的 nsys timeline:
  kernel1 (conv)
  kernel2 (relu)       ← 串行执行
  kernel3 (pool)       ← 一个完了下一个才开始
  kernel4 (conv)
  ...

  问题：GPU 有 20 个 SM，但每个时刻只有 1 个 kernel 在用它们

  能不能让 conv1 和 conv2 同时跑？
  能不能在 GPU 算的时候，CPU 同时准备下一批数据？

  → 这就是 CUDA Streams 解决的问题。

Step 01 bw_bench:
  你测到 BW = 219 GB/s (纯 copy)
  但如果 compute 和 copy 重叠，等效带宽会更高

  比如：
    纯 copy:     BW = 219 GB/s
    compute + copy 重叠:  等效带宽 ≈ BW_copy + BW_compute
    因为 copy 和 compute 用了不同的硬件单元

Step 03 tiled matmul:
  内层循环：每次从显存读一块到 SMEM，在 SMEM 里算
  流水线版：读下一块的同时，算当前块
  → 这就是 stream 的思想：让不同单元同时工作

ADVANCED/01 CUDA Graph:
  Graph 可以包含多个 stream 的操作
  Graph 内的 kernel 可以在不同 stream 上并行
```

## 1. CUDA 操作默认是异步的吗？

```
kernel launch:         异步 (CPU 不等待 kernel 完成就返回)
cudaMemcpy (host↔device):  同步 (CPU 等待传输完成)
cudaMemcpyAsync:        异步 (需要 stream)
cudaMalloc/cudaFree:    同步

直觉理解：
  启动 kernel 很快 (~5 µs)，CPU 不需要等
  但 memcpy 要搬运大量数据，CPU 如果不等，下一行代码可能读到空数据
  所以默认 memcpy 是同步的
```

## 2. Stream 是什么？

```
Stream = GPU 上的独立命令队列

默认情况下，所有操作都在 default stream 上串行执行：

  default stream:  [kernel1] → [kernel2] → [kernel3] → [memcpy] → ...
                   │          │          │            │
                   └─ 全部串行 ──────────┘

多个 stream 可以并行执行：

  stream 1:  [kernel1] ──────→ [kernel3] ──→ ...
  stream 2:       └── [kernel2] ──→ [memcpy] ──→ ...

                  ↑ kernel1 和 kernel2 同时执行
                    利用 GPU 的不同硬件单元
```

### Stream 的硬件基础

```
GPU 内部有多个引擎（engine）：
  - 计算引擎 (SM): 执行 CUDA kernel
  - 拷贝引擎 (Copy): 执行 memcpy (host↔device, device↔device)
  - 视频引擎: 解码/编码

不同的 stream 可以用不同的引擎！
  所以 kernel 和 memcpy 可以重叠
```

## 3. 默认 stream 的陷阱

```
// 这段代码看起来是并行的，但其实不是！
kernel1<<<grid, block>>>();      // 同步操作1
cudaMemcpy(dst, src, n, ...);    // 同步操作2 （等待 kernel1 完成！）

为什么？
  CUDA 的 "default stream" 有两种模式：

  1. Legacy default stream:
     - 所有没有指定 stream 的操作都在同一个 stream
     - 这个 stream 会和其他没指定 stream 的操作同步！
     - 也就是说：kernel1<<<>>>() 会等待之前所有操作完成

  2. Per-thread default stream (推荐):
     - 每个线程有自己的 default stream
     - nvcc --default-stream per-thread
     - 不和其他线程的 stream 同步

建议：总是用 per-thread default stream
```

## 4. 创建和使用 Stream

### CUDA C++

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在 stream1 上启动 kernel
kernel1<<<grid, block, 0, stream1>>>(args...);

// 在 stream2 上启动 kernel (可以和 kernel1 并行!)
kernel2<<<grid, block, 0, stream2>>>(args...);

// 异步 memcpy (必须用 cudaMemcpyAsync)
cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream1);

// 等待所有 stream 完成
cudaDeviceSynchronize();

// 或只等一个 stream
cudaStreamSynchronize(stream1);

// 清理
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### PyTorch

```python
import torch

# 创建 stream
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# 在 stream1 上执行
with torch.cuda.stream(s1):
    y1 = model(x1)  # 这个 forward 在 s1 上执行

# 在 stream2 上执行 (和 s1 并行)
with torch.cuda.stream(s2):
    y2 = model(x2)

# 等两个 stream 都完成
torch.cuda.synchronize()
```

## 5. 重叠模式 1: Compute + Copy

```
最常用的模式：数据搬运和计算重叠。

场景：连续做多次推理
  1. CPU 准备输入数据
  2. 拷贝到 GPU
  3. GPU 推理
  4. 结果拷回 CPU
  5. 回到 1

不用 stream:
  [copy_in] → [compute] → [copy_out] → [copy_in] → [compute] → ...
  ↑ GPU 大部分时间在等数据

用 stream:

  stream1 (copy):  [copy_in]          [copy_out]         [copy_in]
                   └─────── 重叠 ────┐
  stream2 (compute):        [compute]          [compute]
                                    ↑
                                    copy 和 compute 同时进行

  总时间 ≈ max(copy_time, compute_time)
  而不是 copy_time + compute_time
```

### 数据依赖要注意

```
// 错误：kernel 在 copy 完成前就启动了
cudaMemcpyAsync(d_input, input, n, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d_input, ...);
//   ↑ 同一 stream 内按顺序执行，所以这没问题！

// 不同 stream 需要同步！
cudaMemcpyAsync(d_input, input, n, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_input, ...);
//   ↑ stream2 不知道 stream1 在 copy，可能读到不完整的数据！

// 需要用 event 同步
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream1);  // stream1 copy 完后记录 event
cudaStreamWaitEvent(stream2, event);  // stream2 等 event 再执行
kernel<<<grid, block, 0, stream2>>>(d_input, ...);
```

## 6. 重叠模式 2: Pipeline

```
把一个大任务分成小块，轮流在多个 stream 上处理。

和 Step 03 tiled matmul 的流水线完全一样：
  不用流水线: 读块0 → 算块0 → 写块0 → 读块1 → 算块1 → 写块1 → ...

  用流水线:
    stream0:  读块0      算块0      写块0
    stream1:      读块1      算块1      写块1
    stream2:          读块2      算块2      写块2

           ↑ 同时有 2-3 个阶段在运行
```

### 适用场景

```
大矩阵乘法（M, N, K 都很大）：
  分成多个 tile，每个 tile 在不同 stream 上计算
  需要确保没有 shared memory 冲突

多输入推理：
  同时处理 batch=1 的多个输入
  每个输入一个 stream
  总吞吐 ≈ batch_size × single_stream_throughput

数据加载 + 推理：
  stream1: 加载数据 (memcpy H2D)
  stream2: 推理 (kernel)
  交替进行，GPU 不用等数据
```

## 7. Stream 同步

### 同步方法

```
方法 1: cudaDeviceSynchronize()
  等待所有 stream 上的所有操作完成
  最慢，但最简单

方法 2: cudaStreamSynchronize(stream)
  只等一个 stream
  其他 stream 不受影响

方法 3: cudaEvent (stream 间同步)
  cudaEventRecord(event, stream_src)
  cudaStreamWaitEvent(stream_dst, event)
  stream_dst 等待 stream_src 完成某个操作后再继续
  不会阻塞 CPU

方法 4: cudaStreamWaitEvent(event)
  CPU 层面的等待 (阻塞 CPU 线程)
```

### 同步的开销

```
cudaDeviceSynchronize:    ~5-100 µs (取决于 GPU 是否空闲)
cudaStreamSynchronize:    ~3-50 µs
cudaEvent:                ~1 µs (不阻塞 CPU)
```

## 8. CUDA Graph + Streams

```
CUDA Graph 可以包含多个 stream 的操作。

在 capture 期间，你可以切换 stream：
  cudaStreamBeginCapture(stream, mode);

  // stream1 上的操作
  kernel_a<<<grid, block, 0, stream1>>>(...);
  kernel_b<<<grid, block, 0, stream1>>>(...);

  // stream2 上的操作 (和 stream1 并行)
  kernel_c<<<grid, block, 0, stream2>>>(...);

  // 同步
  cudaEventRecord(event, stream1);
  cudaStreamWaitEvent(stream2, event);

  cudaStreamEndCapture(stream1, &graph);

一次 replay = 复现所有 stream 上的操作 + 同步关系
```

## 9. 什么时候用 Stream

### 明显加速

```
1. 数据加载 + 推理交替
   输入在 CPU 准备 → memcpy 到 GPU → 推理 → 结果 memcpy 回 CPU
   stream1: [memcpy_in1] [infer1] [memcpy_out1] [memcpy_in2] [infer2] ...
   stream2:                      [memcpy_in2] [infer2]
   → 准确率调优 (overlap copy and compute)

2. 多输入独立推理
   同时处理多个 batch=1 的请求
   每个请求一个 stream

3. 大模型跨层流水线
   不同层在不同 stream 上
   (需要分析层间依赖)
```

### 效果不大的场景

```
1. 单 kernel 已经占满 GPU (100% SM 占用)
   再加 stream 也没资源了

2. 小模型推理 (< 10 kernels, 每个 < 50 µs)
   stream 的调度开销可能超过收益

3. 内存带宽已经饱和
   加 stream 只会增加带宽竞争
```

## 10. 和前面步骤的连接

```
Step 04 mini_cnn:
  nsys 时间线上所有 kernel 串行
  如果用 stream，conv1 和 conv2 可以部分重叠
  但 mini_cnn 太小，收益有限

Step 03 tiled matmul:
  内层循环的流水线 = stream 的微缩版
  流水线 tiling 在 SMEM 层面
  stream 在 GPU 全局层面

Step 01 bw_bench:
  理论带宽 219 GB/s
  stream 重叠可以让等效带宽更高

ADVANCED/01 CUDA Graph:
  Graph 能捕获多 stream 操作
  Graph + Stream = 最终形态
```

## 11. 总结

```
Stream = 独立命令队列
多个 stream 可以并行执行

关键模式：
  compute + copy 重叠
  流水线多块处理

适用：
  大模型推理
  多输入服务
  计算和数据加载交替

和本课程的关系：
  Step 04 的 nsys timeline 上全是串行
  Stream 是让时间线"变厚"的工具
  你的 kernel 写得再快，如果 GPU 有一半时间在等数据，也没用
```
