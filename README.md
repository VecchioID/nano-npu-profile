# nano-npu-profile: 端侧 NPU Profiling & 性能调试

> 基于 NVIDIA Thor (CC 11.0) 的端侧 SoC Profiling 方法论实战。方法论通用所有端侧 NPU（RK3588、Qualcomm Hexagon、Apple Neural Engine 等）。

## 快速开始

```bash
# 先看前置知识
less 00_PREREQUISITE/TUTORIAL_CUDA.md

# 编译所有 C++ demos
make all

# 按序运行 Steps 01-05
make run
```

## 学习路径

参见 **[ROADMAP.md](ROADMAP.md)** 了解完整的学习路线图。

## 项目结构

```
00_PREREQUISITE/        CUDA 从零教程 + PyTorch→kernel 链路
01_baseline_bandwidth/  带宽 & 算力天花板 (BW=219 GB/s, FMA=6.5 TFLOPS)
02_roofline/            Roofline 模型 (含 roofline.png)
03_operator_deepdive/   算子优化 naive→tiled→WMMA
04_model_level/         Mini CNN 端到端 profiling
05_bottleneck/          瓶颈分类与优化实战
06_Triton/              Python Triton 版 (Steps 01-05 平替)
CASE_STUDY_alexnet/     AlexNet 真实 profiling 案例
ADVANCED/               高级专题 (CUDA Graph, compile, FA, ...)
tools/                  profiling 工具封装
```

## 核心步骤

| 步骤 | 内容 | 关键产出 |
|------|------|---------|
| 00 | CUDA 前置知识 | GPU 架构、kernel 是什么、怎么替换 |
| 01 | 带宽 & 算力天花板 | BW_peak=219 GB/s, FMA=6.5 TFLOPS |
| 02 | Roofline 模型 | roofline.png，各 kernel 的 AI 和性能位置 |
| 03 | 算子优化 naive→tiled→WMMA | 1024³ matmul: 580 → 866 → 3955 GFLOP/s |
| 04 | Mini CNN 端到端 profiling | 3700 inf/s, nsys 时间线, 逐层分析 |
| 05 | 瓶颈分类与优化实战 | memory-bound / compute-bound / 低 occupancy |

## 实战案例

`CASE_STUDY_alexnet/alexnet_profile.py` 对真实 AlexNet 进行逐层 profiling：
- nsys 确认 cuBLAS GEMV (FC 层) 占 GPU 时间 92.4%
- FC1 (AI=0.31) 和 FC2 (AI=0.47) 均为 memory-bound
- Conv 层 (AI=78-316) 通过 cuDNN TF32 Tensor Cores 达到 compute-bound

## 高级专题

| 专题 | 解决的问题 | 前置知识 |
|------|-----------|---------|
| CUDA Graph | 小 batch 推理的 kernel 启动开销 | Step 04 mini_cnn |
| torch.compile | 自动 kernel fusion + 部署加速 | Step 04 / 06_Triton |
| FlashAttention | 长序列 attention 的 IO 瓶颈 | Step 02 roofline |
| Quantization | INT8/FP8 推理加速 | Step 01-02 天花板 |
| CUDA Streams | 计算和传输重叠 | Step 04 nsys 分析 |

## 方法论（通用）

```
1. 定基线         BW_peak + GFLOP_peak
2. 算子分析       每个算子的 arithmetic intensity
3. 模型 profiling  逐层耗时分布
4. 瓶颈分类       根据 roofline 位置分类
5. 针对性优化      → compute-bound: 降精度 / Tensor Core
                  → memory-bound: 数据复用 / 访存合并
                  → low occupancy: 调整 grid/block 配置
6. 验证迭代       重新 profiling 确认瓶颈是否转移
```

## 环境

- **GPU**: NVIDIA Thor (Compute Capability 11.0)
- **CUDA**: 13.0
- **架构**: ARM64
