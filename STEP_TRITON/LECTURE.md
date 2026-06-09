# STEP_TRITON：用 Python 写 GPU Kernel

## 为什么需要 Triton？

CUDA C++ 性能最好，但开发效率低。每个 kernel 要手动管理：
- thread/block 映射
- shared memory 分配和同步
- warp 内规约（shuffle/sync）
- 合并访存（coalescing）
- 寄存器压力

**Triton 帮你自动处理这些。** 你只需要描述"每个 block 做什么"。

## 本目录结构

| 文件 | 对应 C++ 步骤 | 内容 |
|------|-------------|------|
| `00_tutorial.py` | — | Triton 入门教程：语法、概念、对照表 |
| `01_baseline.py` | Step 01 | 测带宽、测 FP32/Tensor Core 吞吐 |
| `02_roofline.py` | Step 02 | roofline 数据点 + 自动画图 |
| `03_matmul.py` | Step 03 | naive → tiled → tl.dot 三层 matmul |
| `04_mini_cnn.py` | Step 04 | 用 Triton kernel 实现的 CNN |
| `05_bottleneck.py` | Step 05 | memory-bound / compute-bound / strided / occupancy 分析 |

## 与 CUDA C++ 的对比

```
概念              CUDA C++                      Triton
─────             ───────                       ──────
block id          blockIdx.x                    tl.program_id(0)
thread id         threadIdx.x                   tl.arange(0, BLOCK)
block dim         blockDim.x                    BLOCK: tl.constexpr (编译期)
grid dim          gridDim.x                     grid = (N,) 元组
访存              ptr[idx]                      tl.load(ptr + offsets, mask=mask)
Shared Memory     __shared__ float s[128];      在 kernel 内定义局部数组即可
同步              __syncthreads()               通常不需要（Triton 自动管理）
Tensor Core       wmma::mma_sync                tl.dot(a, b)
规约              手写 warp shuffle              tl.sum(x, axis=...)
Autotuning        手写多版本                     @triton.autotune(configs=[...])
```

## 运行所有测试

```bash
cd STEP_TRITON

python3 00_tutorial.py    # 入门教程
python3 01_baseline.py    # 测天花板
python3 02_roofline.py    # 画 roofline（生成 roofline_triton.png）
python3 03_matmul.py      # matmul 三层对比
python3 04_mini_cnn.py    # CNN 推理
python3 05_bottleneck.py  # 瓶颈分析
```

## 性能对比预期

| 测试 | C++ 版 | Triton 版 | 预期 |
|------|--------|-----------|------|
| 带宽 copy | ~219 GB/s | ~200-219 GB/s | 接近（受 BLOCK 大小影响） |
| FP32 FMA | ~6.5 TFLOPS | ~5-6.5 TFLOPS | 略低（Triton 有少量开销） |
| Matmul 1024³ | ~3,955 GFLOPS (WMMA) | ~3,000-4,000 GFLOPS | 接近（tl.dot 也用 Tensor Core） |
| Mini CNN | ~3,700 inf/s | ~2,000-4,000 inf/s | 取决于 kernel 优化程度 |

Triton 的性能通常能达到手写 CUDA 的 90-100%，但代码量减少 3-5 倍。
