# nano-npu-profile: 端侧 NPU Profiling & 性能调试

> 基于 NVIDIA Thor (CC 11.0) 的端侧 SoC Profiling 方法论实战。方法论通用所有端侧 NPU（RK3588、Qualcomm Hexagon、Apple Neural Engine 等）。

## 背景

**NVIDIA Thor** 是面向自动驾驶的端侧 SoC，含 GPU + DLA（Deep Learning Accelerator）。它是端侧 NPU 的典型代表：**资源受限、功耗敏感、延迟关键**。本项目教你如何用 profiling 工具定位性能瓶颈，并指导优化方向。

## 学习路径

```
Step 1: Baseline      了解 NPU 的带宽和算力天花板（roofline 模型的 axes）
Step 2: Roofline      构建 roofline 模型 → 可视化 kernel 处于哪一端
Step 3: Operator      深入 Conv2D / MatMul 各种配置的性能特征
Step 4: Model Level   端到端 CNN 推理 profiling + nsys/ncu 采集
Step 5: Bottleneck    从 profiling 数据分类瓶颈 → 针对性优化
```

## 项目结构

```
nano-npu-profile/
├── 01_baseline_bandwidth/   内存带宽 & 计算吞吐基准
│   ├── bw_bench.cu             带宽基准（不同数据量）
│   └── compute_bench.cu        算力基准（FMA / MAC）
├── 02_roofline/              Roofline 模型
│   ├── roofline_gen.cu         生成 roofline 数据点
│   └── plot_roofline.py        可视化 roofline 图
├── 03_operator_deepdive/    算子级分析
│   ├── conv_profile.cu         Conv2D（naive vs preload）
│   └── matmul_profile.cu       MatMul（naive vs tiled, shape 扫描）
├── 04_model_level/          模型级 Profiling
│   ├── mini_cnn.cu             小型 CNN（Conv→ReLU→Pool→FC）
│   └── profile_end2end.py      自动化 ncu/nsys 采集 + 报告
├── 05_bottleneck/           瓶颈定位
│   ├── bound_analysis.cu       5 种瓶颈场景 + 诊断指南
│   └── fix_perf.cu             从 profiling 到优化的完整案例
├── tools/                    Profiling 工具
│   ├── run_ncu.sh              ncu 启动封装
│   ├── run_nsys.sh             nsys 启动封装
│   └── analyze.py              ncu 结果解析 + 自动分类
├── Makefile
└── README.md
```

## 快速开始

```bash
cd nano-npu-profile
make all          # 编译所有 demos
make run          # 运行所有基准测试
```

## Demo 详解

### 01: 内存带宽 & 计算吞吐基准

理解 NPU 的两个天花板：**内存带宽（GB/s）** 和 **计算吞吐（GFLOP/s）**。

```bash
./01_baseline_bandwidth/bw_bench
./01_baseline_bandwidth/compute_bench
```

| 测试项 | 揭示的问题 |
|--------|-----------|
| 不同数据量的带宽 | launch overhead 稀释 vs 带宽饱和 |
| 不同迭代次数的算力 | compute-bound 时的峰值 FLOPS |
| FMA MAC 测试 | 实际可达算力 vs 理论值 |

**产出**: 你的 NPU 的 `BW_peak` 和 `GFLOP_peak` — 这是 roofline 模型的基础。

### 02: Roofline 模型

Roofline 是最直观的性能分析方法论。每个 kernel 有 `Arithmetic Intensity (FLOP/Byte)`，落在 roofline 图上的位置决定瓶颈：

```
GFLOP/s ↑
        | ─────────── compute ceiling (水平线)
        |          ╱
        |  ●      ╱  compute-bound
        |  matmul ╱
        |        ╱
        |  ●    ╱  memory-bound
        | copy  ╱
        |      ╱  memory ceiling (斜率 = BW_peak)
        └──────────────────→ FLOP/Byte
              ridge point
```

```bash
./02_roofline/roofline_gen          # 生成各 kernel 的 AI + 性能
python3 plot_roofline.py            # 可视化
```

### 03: 算子级 Profiling

深入了解关键算子的性能特征：

```bash
./03_operator_deepdive/matmul_profile
```

- **Shape 扫描**: 不同 M/N/K 组合的 matmul 性能
- **Naive vs Tiled**: shared memory tiling 的真实加速比

```bash
./03_operator_deepdive/conv_profile
```

- **Conv2D 配置扫描**: 不同 H/W/C/K 组合
- **Naive vs Preload**: 数据复用 vs 重复读取

### 04: 模型级 Profiling

端到端 CNN 推理的完整 profiling 流程：

```bash
./04_model_level/mini_cnn           # 运行推理 + 基本计时
python3 profile_end2end.py          # ncu 深度 profiling + 报告
python3 profile_end2end.py --nsys   # nsys 时间线分析
```

**架构**: Conv(3x3, 3→16) → ReLU → MaxPool(2x2) → Conv(3x3, 16→32) → ReLU → FC(→10)

**nsys 时间线**可以清晰看到：
- 每个 kernel 的 launch 间隔
- CPU-GPU 同步点
- 各算子的执行顺序和重叠

### 05: 瓶颈定位实战

从 profiling 数据到优化决策：

```bash
./05_bottleneck/bound_analysis  # 5 种瓶颈场景
./05_bottleneck/fix_perf        # 完整优化案例
```

| ncu 指标 | 分类依据 |
|----------|---------|
| SM Busy > 60%, Mem Pipes < 30% | Compute-bound → 降精度 / Tensor Core |
| Mem Pipes > 60%, SM Busy < 30% | Memory-bound → tiling / coalescing |
| SM Busy + Mem Pipes 双低 | Low occupancy → 增大 block / 减少 shared |
| 两者相近 | Balanced → 两方面都需优化 |

## 工具链

### ncu (Nsight Compute) — 单 kernel 微观分析

```bash
bash tools/run_ncu.sh ./04_model_level/mini_cnn
```

关键指标：
- `Achieved Occupancy`: SM 利用率
- `SM Busy`: 计算单元忙碌比例
- `Mem Pipes Busy`: 访存单元忙碌比例
- `DRAM Throughput`: 实际带宽
- `L1/L2 Hit Rate`: 缓存命中率

### nsys (Nsight Systems) — 全模型时间线分析

```bash
bash tools/run_nsys.sh ./04_model_level/mini_cnn
nsys-ui profile_mini_cnn_*.nsys-rep
```

关键视图：
- CUDA Kernel timeline: 各 kernel 执行顺序
- CPU-GPU 同步: cudaDeviceSynchronize 的代价
- Memory ops: memcpy/memset 的带宽

### analyze.py — 自动分析

```bash
python3 tools/analyze.py ./04_model_level/mini_cnn
```

自动输出瓶颈分类 + 优化建议。

## 端侧 NPU Profiling 方法论（通用）

```
1. 定基线         测 BW_peak + GFLOP_peak
2. 算子分析       每个算子的 arithmetic intensity
3. 模型 profiling  逐层耗时分布
4. 瓶颈分类       根据 roofline 位置分类
5. 针对性优化      compute-bound → 降精度 / Tensor Core
                    memory-bound → 数据复用 / 访存合并
                    low occupancy → 调整 grid/block 配置
6. 验证迭代       重新 profiling 确认瓶颈是否转移
```

## 环境

- **GPU**: NVIDIA Thor (Compute Capability 11.0)
- **CUDA**: 13.0
- **架构**: ARM64

## 参考

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Roofline Model (Berkeley)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/roofline.pdf)
- [NVIDIA Thor 平台](https://www.nvidia.com/en-us/automotive/drive/thor/)
