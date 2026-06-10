# 02: torch.compile — 自动优化你的模型

## 0. 这条线是怎么连起来的

```
Step 03 matmul: 你手写了三个版本
  naive:  580 GFLOP/s   (3 个 kernel: read A, read B, write C)
  tiled:  866 GFLOP/s   (把 A/B 切块放到 shared memory)
  WMMA:  3955 GFLOP/s   (Tensor Core)

  核心思路：减少访存 = 加速
  naive 的问题：每次计算都从显存读，算完写回，下次再用又读
  tiled 的优化：把数据搬到 shared memory，一块一块算
  **torch.compile 做的就是这件事——自动的**

Step 04 mini_cnn: 6 个 C++ kernel
  conv1 → relu1 → pool1 → conv2 → relu2 → fc
  每个 kernel：从显存读 → 算 → 写回显存
  conv1 的输出写回显存，relu1 再读回来
  这中间的读写完全可以省掉！
  **torch.compile 做的就是这件事——kernel fusion**

Step 05 的 bottleneck 分析：
  memory-bound: 带宽利用率低
  为什么低？因为每个 kernel 只做一点点事就写回显存
  如果 kernel 做大一点（融合），带宽利用率自然上去了
  **torch.compile 做的就是这件事——提高 AI**

Step 06 Triton LECTURE.md 的对照表：
  Triton 自动管理 shared memory、自动 coalescing
  torch.compile 的后端 Inductor 就靠 Triton 生成 fused kernel
  **torch.compile 就是调用 Python 版的 Step 06**

01_CUDA_Graph:
  CUDA Graph 消除了 kernel 间的启动开销
  torch.compile(mode="reduce-overhead") 自动用 CUDA Graph
  **torch.compile = kernel fusion + CUDA Graph 的合体**
```

## 1. 一句话总结

```python
model = torch.compile(model)
```

这一行 = 自动帮你做你在 Steps 01-06 手写的一切优化：

```
你的 Step 03 tiled matmul:
  – 手动分 tile
  – 手动搬 shared memory
  – 手动同步

torch.compile:
  – 自动分析计算图
  – 自动做 tiling
  – 自动 fused kernel
  – 自动用 Tensor Core (tl.dot)
  – 自动 autotune 选最好的 tile 大小
  – 自动 CUDA Graph
  — 你什么都不用管
```

## 2. 三种模式

| mode | 做了什么 | 适合 |
|------|---------|------|
| `default` | kernel fusion + Triton 编译 | 通用 |
| `reduce-overhead` | kernel fusion + Triton + **CUDA Graph** | 小 batch 推理 |
| `max-autotune` | 上面 + 穷举所有 tile 配置 | 部署（编译一次跑很久） |

**`reduce-overhead` = 把 ADVANCED/01 的 CUDA Graph 自动打开了。**

## 3. Kernel Fusion 到底做了什么

回到 Step 03 你写的 naive matmul：

```python
# naive matmul 的执行流程：
for i in range(M):
    for j in range(N):
        c[i][j] = sum(a[i][k] * b[k][j] for k in range(K))

# GPU 上的实际执行（naive）：
kernel 1: 读 A 的一行 → 算 dot product → 写 C 的一个元素
           ← 你手动做了 tiling 来解决这个问题

# mini_cnn 的 pipeline（Step 04）：
kernel 1: conv1       读 X → 算 → 写 X1
kernel 2: relu        读 X1 → 算 → 写 X2
kernel 3: pool        读 X2 → 算 → 写 X3
kernel 4: conv2       读 X3 → 算 → 写 X4
kernel 5: relu        读 X4 → 算 → 写 X5
kernel 6: fc          读 X5 → 算 → 写 Y

总共：6 次读 + 6 次写 = 12 次 DRAM 访问

# torch.compile 后： 
fused_kernel: conv1+relu+pool+conv2+relu+fc
              读 X → 算 → 写 Y
              只需要 1 次读 + 1 次写 = 2 次 DRAM 访问

# 这会带来什么效果？
```

### 效果 1：减少 DRAM 带宽压力

回到 Step 02 的 roofline — 看 memory ceiling：

```
原始 6 个 kernel（每个都单独读写显存）:
  总数据量 = 6 × (输入 + 输出) = 很大
  带宽利用率低，因为数据在 DRAM 和 SM 之间反复搬运

fused 1 个 kernel（中间结果在寄存器/SMEM，不进 DRAM）:
  总数据量 = 1 × (输入 + 输出) = 小很多
  更高 arithmetic intensity → 在 roofline 上往右移动

  DRAM 带宽瓶颈 → 最多 219 GB/s
  fusion 后：中间结果不经过 DRAM，等效带宽更高
  → 这就是 Step 03 tiling 的核心思想：数据复用
```

### 效果 2：减少启动开销

```
原始：6 次 kernel launch × ~5 µs = 30 µs
fused：1 次 kernel launch = 5 µs
省了 25 µs

回到 Step 04 mini_cnn：
  原始 3700 inf/s
  省了 25 µs 启动 + 若干 DRAM 读写
  预计 fused 后 ~4500-5000 inf/s
```

## 4. torch.compile 的三层架构

```
                    ┌─────────────────────────┐
  你的 Python 代码 → │ TorchDynamo (图形捕获)  │
                    │ 拦截 forward 执行        │
                    │ 记录为计算图 (FX Graph)  │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Inductor (编译后端)      │
                    │ 分析计算图               │
                    │ 做 kernel fusion 决策    │
                    │ 生成 Triton kernel 代码  │
                    │ 自动 autotune            │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Triton kernel (TMA)      │
                    │ 在 GPU 上执行            │
                    └─────────────────────────┘
```

注意最后一层：**torch.compile 生成的 fused kernel 就是用 Triton 写的。** 你已经在 Step 06 学了 Triton 语法，所以你现在看得懂 torch.compile 在干什么。

## 5. Fusion 能做什么，不能做什么

### 能 fused

```
conv + bn + relu        → conv_bn_relu_kernel
linear + relu           → linear_relu_kernel
add + relu              → residual_kernel
element-wise 序列       → element_fused_kernel

记忆口诀：连续的数据变换可以合并
         因为中间结果只需要存在寄存器里
         → 这和 Step 03 tiled matmul 的思路一模一样
```

### 不能 fused

```
reshape / transpose / view（需要改变数据布局）
gather / scatter（不规则访存）
MaxPool / AvgPool（需要特殊硬件单元）

记忆口诀：需要全局数据重排的不能合并
         → 因为要写回显存才能做 reshape
```

## 6. 和前面步骤的具体连接

### 连接 Step 03（matmul 优化）

```
你在 Step 03 手动写了：
  naive:  每个线程算一个 C 元素 → 3 次全局内存访问/FLOP（580 GFLOPS）
  tiled:  把 A/B 分块到 shared memory → 1 次全局内存访问/很多 FLOP（866 GFLOPS）
  WMMA:  Tensor Core 硬件加速 → 一步算 16×16×16（3955 GFLOPS）

torch.compile 自动做：
  tiling → Triton 后端自动选最优 tile 大小
  Tensor Core → tl.dot 自动映射到 WMMA
  autotune → 试遍所有 tile 配置，选最快的

你可以理解成：
  torch.compile = 自动化的 Step 03
  而且它还能做跨操作的 fusion（比如 conv+relu）
```

### 连接 Step 02（roofline）

```
原始操作：
  conv:   AI=~200 (compute-bound)
  relu:   AI=~0.1 (memory-bound, 因为只做了一个 element-wise 就写回)
  pool:   AI=~2   (memory-bound)

fused 操作：
  conv+relu+pool: AI ≈ conv 的 AI + 中间结果不再进出 DRAM
                   → 等效 AI 更高
                   → 在 roofline 上更往右

为什么？
  不 fused：conv 输出写 DRAM → relu 从 DRAM 读 → pool 写回 DRAM
  fused：conv 输出在寄存器 → relu 直接算 → pool 直接算 → 写 DRAM
         省了 conv→relu、relu→pool 两次 DRAM 读写
```

### 连接 06_Triton

```
你在 Step 06 学了 Triton kernel 怎么写。
torch.compile 的 Inductor 后端，生成的代码就是 Triton kernel。

换句话说：
  你在 Step 06 手写的 matmul_kernel：
    @triton.jit
    def matmul_kernel(A, B, C, ...):
        pid = tl.program_id(0)
        ...

  torch.compile 的 Inductor 自动生成类似的代码：
    def fused_kernel(x, w1, b1, w2, b2):
        # 自动生成的 Triton kernel
        h = tl.load(x + offsets)
        h = tl.dot(h, W1)    # ← 这就是你 06_Triton/03 学的
        h = h + b1
        h = max(h, 0)        # relu
        h = tl.dot(h, W2)
        ...
```

### 连接 AlexNet case

```
alexnet_profile.py 的 nsys 数据：
  FC 层占 92.4% GPU 时间
  每个 FC 层 = 一个很小的 cuBLAS GEMV kernel
  每层之间：读显存 → 算 → 写显存 → 下个 kernel 再读

torch.compile 对 AlexNet 的优化：
  FC1 + relu → 1 个 fused kernel（省 1 次中间结果读写）
  FC2 + relu → 1 个 fused kernel
  FC3 + relu → 1 个 fused kernel
  FC4        → 1 个 kernel
  
  原来：8 个 FC kernel + 3 个 relu kernel = 11 个 kernel
  fused：4 个 kernel
  
  省掉：7 次 launch + 3 次 DRAM 中间结果读写
  batch=1 预计加速 +20-30%
```

## 7. TensorRT vs torch.compile

| | torch.compile | TensorRT |
|--|-------------|----------|
| 核心技术 | Triton kernel fusion + 自动 tiling | 专有 engine，更强 |
| 代码改动 | 一行 `torch.compile(model)` | 需要 ONNX 导出 |
| 动态 shape | 原生支持 | 需要配置 |
| 部署依赖 | 需要 PyTorch | 独立 SDK |
| 量化 | 实验性 | 成熟（INT8/FP8） |
| 性能提升 | +20-100% | +50-300% |

**选择策略（从简单到复杂）：**

```
1. model = torch.compile(model)        # 零成本，先加上
2. model = torch.compile(mode="reduce-overhead")  # 小 batch 再用
3. torch_tensorrt.compile(model, ...)  # 还不够，上 TensorRT
4. 手写 C++ kernel + CUDA Graph       # 还不行，回到 Step 03-05
```

## 8. 总结

```
你在 Steps 01-06 学到的每件事，都指向同一个终点：

Step 01: 测天花板 → compile 帮你接近天花板
Step 02: roofline → compile 提高你的 AI（fusion）
Step 03: matmul 优化 → compile 自动 tiling + Tensor Core
Step 04: mini_cnn profiling → compile 减少 kernel 数量
Step 05: 瓶颈分类 → compile 同时解决三类瓶颈
Step 06: Triton → compile 的后端就是你写的 Triton kernel
CUDA Graph → compile 的 reduce-overhead 模式自动启用

下一章：03_FlashAttention
  — attention 的 roofline 分析 + IO-aware 优化
  你会发现又是在用 Step 02 的 roofline 和 Step 03 的 tiling
```
