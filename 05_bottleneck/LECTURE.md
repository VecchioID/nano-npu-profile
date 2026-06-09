# Step 05: Bottleneck Classification & Fix

## 本步目标

掌握如何诊断一个 CUDA kernel 的瓶颈类型，以及对应的优化手段。

```
一个 kernel 慢，无非三种情况：
  ① 等数据（memory-bound）  → 提高数据复用 / 改善访存模式
  ② 等计算（compute-bound） → 减少计算量 / 用更低精度
  ③ 空转（occupancy 低）    → 更多线程 / 更多 block
```

---

## 第一部分：bound_analysis — 五种瓶颈场景

### 编译运行

```bash
make 05_bottleneck/bound_analysis
./05_bottleneck/bound_analysis
```

### 结果

```
mem-bound (1 read + 1 write)      0.707 ms
compute-bound (100 iters)        42.104 ms
compute-bound (1000 iters)      416.692 ms
balanced (8 flops per element)    0.822 ms
stride=1 (coalesced)              0.820 ms
stride=4 (bad coalescing)         1.948 ms
stride=32 (worst)                 6.578 ms
low occ (32 thr/block)            1.598 ms
high occ (256 thr/block)          0.704 ms
```

### 场景解读

#### 1. Memory-bound（0.707 ms）

```cuda
out[idx] = in[idx] * 2.0f;  // 1 次读 + 1 次写，0 次计算复用
```

16M 个 float → 读取 64 MB + 写入 64 MB = 128 MB 数据。
时间 0.707 ms → `128 MB / 0.707 ms = 181 GB/s`，接近带宽天花板（219 GB/s）。
**瓶颈在带宽，不在计算。**

#### 2. Compute-bound（42 ~ 417 ms）

```cuda
for (int i = 0; i < iters; i++) {
    x = sinf(cosf(x * x + 0.5f));
    x = expf(logf(fabsf(x) + 1e-6f));
}
```

100 iters → 42 ms, 1000 iters → 417 ms（线性增长）。
几乎没有访存（只读写一次），纯算数学函数。
**瓶颈在计算单元（SFU），不在带宽。**

#### 3. 平衡型（0.822 ms）

```cuda
for (int i = 0; i < 8; i++) x = x * 0.5f + 0.5f;
```

8 次乘加 + 1 次读 + 1 次写 → 计算和访存大致均衡。
时间和纯 memory-bound 差不多（0.707 vs 0.822），说明 8 次乘加几乎不花钱——被访存延迟隐藏了。

#### 4. 非合并访问：stride 的影响（0.82 → 6.58 ms）

```cuda
out[idx] = in[idx * stride];
```

| stride | 时间 | 倍数 |
|--------|------|------|
| 1（合并） | 0.820 ms | 1× |
| 4（不合并） | 1.948 ms | 2.4× |
| 32（最差） | 6.578 ms | 8.0× |

**为什么 stride 大会慢？**

看 warp 内的 32 个线程在 stride=1 时怎么访存：

```
stride=1:
  Thread 0 → 读 addr [0]
  Thread 1 → 读 addr [1]      ← 连续地址 → 硬件合并成一个 128 字节的突发读取
  Thread 2 → 读 addr [2]
  ...（32 个连续地址）

stride=32:
  Thread 0 → 读 addr [0]
  Thread 1 → 读 addr [32]     ← 每个地址隔 32 个 float = 128 字节
  Thread 2 → 读 addr [64]     ← 无法合并 → 每个线程单独发一次访存请求
  ...
```

**合并访问（coalescing）**：当 warp 内线程访问连续内存时，硬件把 32 个请求合并成一次大的突发传输。不连续就拆成 32 次小请求，带宽利用率暴跌。

#### 5. 低占用率（1.598 vs 0.704 ms）

| block 大小 | 时间 |
|-----------|------|
| 32 线程/block | 1.598 ms |
| 256 线程/block | 0.704 ms |

**为什么 32 线程更慢？**

32 线程 = 1 个 warp。SM 的 warp scheduler 只有一个 warp 可以调度。这个 warp 在等数据时（访存延迟 ~200-800 cycle），SM 无事可做。

256 线程 = 8 个 warp。某个 warp 等数据时，scheduler 立刻切换到另一个 warp，**用计算隐藏延迟**。

这就是 **occupancy（占用率）**：活跃 warp 数 ÷ SM 最大 warp 数（Thor 每个 SM 最大 32 个 warp）。warp 越多，SM 越不容易空闲。

---

## 第二部分：fix_perf — Reduction 优化案例

### 三种实现

#### Bad：串行规约

```cuda
if (tid == 0) {
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += sdata[i];
    out[blockIdx.x] = sum;
}
```

只有 1 个线程算和，255 个线程闲着。

#### Good：树形规约

```cuda
for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

所有线程参与：128 → 64 → 32 → ... → 1，总共 log₂(256) = 8 步。

#### Warp：shuffle 规约

```cuda
float val = sdata[tid];
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, offset);
```

用寄存器 shuffle 代替 shared memory，避免 __syncthreads()。

### 实测结果

```
Bad  (serial):     2.546 ms
Good (tree):       3.054 ms   ← 反而更慢
Warp (shuffle):    2.455 ms   ← 基本没差
```

**为什么三种实现几乎一样？**

因为 256 MB 数据从 global memory 读到 shared memory 这一步占了绝大部分时间。256 个元素的加法（不管是串行、树形、还是 shuffle）相对于访存时间来说可以忽略不计。

这个结果本身就是一个教训：**Profile 之前不要假设瓶颈在哪。** 你以为 reduce 部分是瓶颈，其实读数据才是。

---

## 总结

| 瓶颈类型 | 现象 | 本质原因 | 优化方向 |
|---------|------|---------|---------|
| Memory-bound | 耗时≈数据量÷带宽 | 计算太少，等数据 | 提高 AI（tiling、fusing） |
| Compute-bound | 耗时随计算量线性增长 | 运算太多 | 降低精度、用 Tensor Core |
| 非合并访问 | stride 大时数倍变慢 | 缓存行利用率低 | 调整数据布局（AoS→SoA） |
| 低占用率 | block 线程少时更慢 | warp 不够隐藏延迟 | 增大 block 大小 |
| 启动开销 | kernel 耗时 < 10 µs | CPU→GPU 握手 | CUDA Graph / 合并 kernel |

**Step 05 的核心：先分类瓶颈类型，再选对应的优化手段。分类错了，优化就白做了。**
