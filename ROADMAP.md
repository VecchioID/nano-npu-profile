# nano-npu-profile 学习路径

## 路径总览

```
                                  ┌─────────────────────────────────┐
                                  │  00_PREREQUISITE/               │
                                  │  ├── TUTORIAL_CUDA.md (11章)    │
                                  │  │  GPU 架构 → kernel → 链路    │
                                  └─────────┬───────────────────────┘
                                            │
                                            ▼
                ┌────────────────────────────────────────────┐
                │     CORE TRACK (CUDA C++)                  │
                │     GPU 性能分析方法论                      │
                ├────────────────────────────────────────────┤
                │  Step 01 → 基线测定 (BW + FMA 天花板)       │
                │  Step 02 → Roofline 模型                   │
                │  Step 03 → 算子优化 (matmul/conv)          │
                │  Step 04 → 模型级 Profiling (mini CNN)     │
                │  Step 05 → 瓶颈分类与优化实战               │
                └──────┬──────────────────────────────┬─────┘
                       │                              │
                       ▼                              ▼
          ┌──────────────────────┐   ┌──────────────────────────┐
          │ CASE STUDY           │   │ PARALLEL TRACK           │
          │ AlexNet 真实 profiling│   │ (Python Triton)          │
          │ nsys 逐层分析         │   │ 用 Python 复现 Steps 1-5 │
          │ roofline 瓶颈定位    │   │ 代码量少 3-5x，性能 ~90%  │
          └──────────────────────┘   └──────────┬───────────────┘
                                                 │
                                                 ▼
                ┌────────────────────────────────────────────┐
                │     ADVANCED TOPICS                        │
                │     基于核心方法论延伸                      │
                ├────────────────────────────────────────────┤
                │  01_CUDA_Graph    ← 启动开销瓶颈的终极解法  │
                │  02_torch_compile ← 零成本部署加速          │
                │  03_FlashAttention                         │
                │  04_Quantization                           │
                │  05_CUDA_Streams                           │
                └────────────────────────────────────────────┘
```

## 两条学习路线

### 路线 A：C++ CUDA 主线（推荐）

| 步骤 | 内容 | 核心概念 | 产出 |
|------|------|---------|------|
| 00 | PREREQUISITE | GPU 架构、kernel 链路、thread/warp/SM | 理论基础 |
| 01 | Baseline | BW_peak, FMA_peak, float4 copy | 性能天花板 |
| 02 | Roofline | AI, ridge point, roofline 图 | 分析方法论 |
| 03 | MatMul/Conv | naive→tiled→WMMA, preload, shared mem | 算子优化 |
| 04 | Mini CNN | 端到端推理, nsys 时间线, 逐层分析 | 模型 profiling |
| 05 | Bottleneck | memory-bound, compute-bound, occupancy, coalescing | 瓶颈诊断 |
| — | AlexNet Case | 真实网络 profiling, FC = 92.4% 瓶颈 | 实战验证 |
| 07+ | ADVANCED/... | CUDA Graph, compile, FlashAttention... | 进阶 |

### 路线 B：Python Triton 快速上手

如果不想学 C++ CUDA，直接从 `06_Triton/` 开始：

| 文件 | 对应 C++ 步骤 | 核心概念 |
|------|-------------|---------|
| 00_tutorial.py | — | Triton 语法、概念、和 CUDA 对照 |
| 01_baseline.py | Step 01 | BW / FMA 天花板 |
| 02_roofline.py | Step 02 | roofline 自动画图 |
| 03_matmul.py | Step 03 | naive→tiled→tl.dot |
| 04_mini_cnn.py | Step 04 | CNN 推理 |
| 05_bottleneck.py | Step 05 | 6 种瓶颈场景 |

## Steps 01-05 与 ADVANCED 的连接

```
Step 03 → matmul 的 naive (580G) → tiled (866G) → WMMA (3955G)
            ↓
ADVANCED/01_CUDA_Graph
            ↓
          mini_cnn 里每个 kernel 都有启动开销 (~5μs)
          几十个 kernel → 几十 μs 的浪费
          CUDA Graph = 一次 capture，无限 replay
          解决 Step 04 的 launch overhead 瓶颈

Step 04 → mini_cnn 的 6 个 kernel
            ↓
ADVANCED/02_torch_compile
            ↓
          torch.compile 自动做 kernel fusion
          把 6 个 kernel 合并成 2-3 个
          解决 Step 04 的 kernel 过多的问题
          一行代码：model = torch.compile(model)
```

## 关键连接点

| 问题 | 在哪发现的 | 解决方案 | 在哪学 |
|------|-----------|---------|-------|
| Kernel 启动开销大 | Step 04 mini_cnn (3700 inf/s 饱和) | CUDA Graph | ADVANCED/01 |
| 小 batch 推理慢 | AlexNet FC 层 (92.4% GPU time) | torch.compile | ADVANCED/02 |
| 单个 kernel 慢 | Step 03 所有 matmul 结果 | 优化 tiling / Tensor Core | Step 03 |
| 模型融合 | Step 04 多次 kernel launch | torch.compile / 手写融合 | ADVANCED/02 |

## 文件结构

```
nano-npu-profile/
├── 00_PREREQUISITE/          ← 前置知识：CUDA 从零 + PyTorch→kernel 链路
│   └── TUTORIAL_CUDA.md
├── 01_baseline_bandwidth/    ← C++ 基线
├── 02_roofline/              ← C++ roofline (含 roofline.png)
├── 03_operator_deepdive/     ← C++ 算子
├── 04_model_level/           ← C++ mini_cnn
├── 05_bottleneck/            ← C++ 瓶颈分析
├── 06_Triton/                ← Python Triton (Steps 01-05 平替)
│   ├── 00~05_*.py
│   └── LECTURE.md
├── CASE_STUDY_alexnet/       ← AlexNet 实战 profiling
│   ├── alexnet_profile.py
│   └── alexnet_roofline.png
├── ADVANCED/                 ← 高级专题
│   ├── 01_CUDA_Graph/
│   ├── 02_torch_compile/
│   ├── 03_FlashAttention/
│   ├── 04_Quantization/
│   └── 05_CUDA_Streams/
├── tools/                    ← profiling 工具封装
├── Makefile                  ← 编译所有 C++ demos
├── ROADMAP.md                ← ← 就是这个文件
└── README.md                 ← 入口
```
