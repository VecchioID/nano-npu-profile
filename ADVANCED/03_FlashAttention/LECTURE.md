# 03: FlashAttention — 把 Attention 的 roofline 往右推

## 0. 这条线是怎么连起来的

```
Step 02 roofline: 你学会了看 kernel 的 arithmetic intensity (AI)
  AI = FLOP / Byte
  AI 低 = memory-bound（在 roofline 的左边）
  AI 高 = compute-bound（在 roofline 的右边）

Standard Attention 的问题：
  Q (N×d) × Kᵀ (d×N) → S (N×N)     ← 这一步算得快（GEMM）
  S → softmax → P (N×N)              ← 这一步要读写整个 N×N 矩阵
  P × V (N×d) → O (N×d)            ← 再读一次 N×N

  关键：S 和 P 是 N×N 的，要写回 HBM（显存）
  对于 N=4096: N×N = 16M 个元素 = 64 MB（FP32）
  对于 N=8192: N×N = 64M 个元素 = 256 MB

  AI 很低，因为 large N×N 中间结果 = 大量读写

Step 03 tiled matmul: 你把大矩阵切块搬到 shared memory
  思路：不要一次性算完，一块一块算，中间结果留在 SM 里

FlashAttention 的核心思想完全一样：
  不要算完整的 N×N S 矩阵
  把 Q/K/V 分块，一块一块算 attention
  中间结果（softmax 的局部统计量）留在寄存器/SMEM
```

---

## 1. Standard Attention 为什么慢

### 计算过程的数据流

```
                    ┌─────────────────────────────────────────┐
                    │           HBM (显存)                    │
                    │                                         │
                    │  ┌──────┐    ┌──────┐    ┌──────┐      │
                    │  │  Q   │    │  K   │    │  V   │      │
                    │  │ N×d  │    │ N×d  │    │ N×d  │      │
                    │  └──┬───┘    └──┬───┘    └──┬───┘      │
                    │     │           │           │          │
                    └─────┼───────────┼───────────┼──────────┘
                          │           │           │
                          ▼           ▼           │
                    ┌──────────────────┐          │
             步 1   │  S = Q × Kᵀ     │          │
                    │  (N×d)×(d×N)    │          │
                    │  → S (N×N)      │          │
                    │  写回 HBM       │          │
                    └────────┬─────────┘          │
                             │                   │
                             ▼                   │
                    ┌──────────────────┐          │
             步 2   │  P = softmax(S)  │          │
                    │  读 S (N×N)     │          │
                    │  写 P (N×N)     │          │
                    └────────┬─────────┘          │
                             │                   │
                             ▼                   ▼
                    ┌───────────────────────────────┐
             步 3   │  O = P × V                    │
                    │  读 P (N×N) + 读 V (N×d)     │
                    │  写 O (N×d)                  │
                    └───────────────────────────────┘
                              │
                              ▼
                          O (N×d)

HBM 访问路径:
  步 1:  Q ──→ ┐
        K ──→ ─┤  GEMM ──→ S 写回 HBM
               ┘
  步 2:  S ──→ softmax ──→ P 写回 HBM
  步 3:  P ──→ ┐
        V ──→ ─┤  GEMM ──→ O
               ┘

  → HBM 读写: 4N² + 4Nd ≈ 4N² (N >> d 时)
  → N×N 矩阵被读写 3 次 (写 S, 读 S, 写 P, 读 P)
```

### Roofline 分析

```
                                GFLOP/s ↑
                                        │
                                  算力天花板 ───────────────────
                                        │      ● Standard Attention
                                        │      AI = d/2
                                  带宽天花板 ──╲
                                        │     ╲
                                        │      ╲
                                        └──────────────────────→ FLOP/Byte
                                        ridge
                                        point
                                        (AI=~34)

  AI = FLOPs / Bytes
     = (4N²d)       / (8N²)     (FP16)
     = d/2

  对于 d=64:   AI = 32   →  ridge_point 附近
  对于 d=128:  AI = 64   →  可能 compute-bound

  但不管 AI 是多少，你都要读写 N² 次的 HBM！
  这意味着 N 越大，Standard Attention 越慢（O(N²) 增长）
```

### HBM 访问随 N 增长

```
                                        HBM 访问 (FP16)
                    N      |  Standard (N²)    | Flash (Nd)
                    ───────┼──────────────────┼──────────────
                    512    │     1 M 元素      │   128 K 元素
                           │                   │
                    1024   │     4 M           │   256 K
                           │                   │
                    2048   │    16 M  ←─ ●     │   512 K  ←─ ▲
                           │         |         │
                    4096   │    64 M     |     │   1 M
                           │         N² 增长   │
                    8192   │   256 M     |     │   2 M     Nd 增长
                           │         |         │
                    ───────┼──────────────────┼──────────────
                           │  O(N²)          │  O(N)
                           │  N 每翻倍 → 4x   │  N 每翻倍 → 2x
```

---

## 2. FlashAttention 的核心思想

### 分块策略一览

```
Standard Attention:
  Q                      K                      V
  ┌──────────┐           ┌──────────┐          ┌──────────┐
  │          │           │          │          │          │
  │   N×d    │    GEMM   │   N×d    │          │   N×d    │
  │          │ ────────→ S(N×N) ──→ softmax ──→ P(N×N) ──→ O(N×d)
  │          │           │          │          │          │
  └──────────┘           └──────────┘          └──────────┘
                          ↑         ↑
                          全部放入 HBM，再读回来

FlashAttention:
  只搬需要的块到 shared memory，中间结果留在 SM 内

  外层循环 (Q 的行):
    ┌──────────┐
    │ Q_block  │  B_r×d      ← 搬到 SMEM
    │ ──────── │
    │          │
    └──────────┘
         │
         ▼ 内层循环 (K/V 的行):
    ┌──────────┐  ┌──────────┐
    │ K_block  │  │ V_block  │   B_c×d    ← 搬到 SMEM
    │ ──────── │  │ ──────── │
    │          │  │          │
    └──────────┘  └──────────┘
         │              │
         ▼              ▼
    ┌──────────────────────┐
    │ S_block = Q_block    │  在 SMEM 中计算
    │         × K_blockᵀ   │  结果也在 SMEM 中
    │                     │
    │ online softmax      │  不写回 HBM
    │                     │
    │ O += P_block × V_blk│  直接累加到寄存器
    └──────────────────────┘   中的 O_block
```

### 和 Step 03 tiled matmul 完全一样的结构

```
Step 03 tiled matmul:                  FlashAttention:
┌──────────────────────┐              ┌──────────────────────┐
│ for tk in range(K):  │              │ for col in range(N): │
│   load A_block       │              │   load K_block      │
│   load B_block       │              │   load V_block      │
│   C += A×B           │              │   S = Q_block×K_blk │
│                      │              │   online_softmax(S) │
│                      │              │   O += P×V_block    │
└──────────────────────┘              └──────────────────────┘
         │                                       │
         └────────────── 都是 tiling ─────────────┘
```

### Online Softmax 的细节

```
Standard softmax（一次算整行）:
  s = [s₁  s₂  s₃  ...  sₙ]          ← 整行 N 个元素

  m = max(s)
    = max(s₁, s₂, ..., sₙ)           ← 1 次全局规约

  p = exp(s - m)
    = [exp(s₁-m)  exp(s₂-m)  ...]    ← 每个元素算 exp

  l = sum(p)
    = exp(s₁-m) + exp(s₂-m) + ...    ← 1 次全局规约

  softmax = p / l                     ← 最终结果


Online softmax（一块一块算）:

  第 0 块:                           第 1 块:
  ┌────────────────┐                 ┌────────────────┐
  │ s = [s₁  s₂]   │                 │ s = [s₃  s₄]   │
  │                │                 │                │
  │ m₀ = max(s₁,s₂)│                 │ m₁ = max(s₃,s₄)│
  │ p₀ = exp(s-m₀) │                 │ p₁ = exp(s-m₁) │
  │ l₀ = sum(p₀)   │                 │ l₁ = sum(p₁)   │
  └────────────────┘                 └────────────────┘
         │                                   │
         └──────────── 合并 ──────────────────┘
                      │
                      ▼
            ┌────────────────────┐
            │ new_m = max(m₀,m₁) │
            │                    │
            │ new_l = exp(m₀-m₁) │  ← 用指数修正前一块的 sum
            │       × l₀ + l₁   │
            │                    │
            │ O = O × exp(m₀-m₁) │  ← 同样修正 O
            │     + exp(s-m₁)×V  │
            └────────────────────┘

  关键是：max 和 sum 可以增量更新！
  不需要完整的 N×N 矩阵
```

### HBM 访问对比：Standard vs Flash

```
Standard Attention:
  时间 ──────────────────────────────────→
       |  QKᵀ   | softmax |  PV    |
       |────────|─────────|────────|
  HBM:  R Q K   W S   R S W P   R P V  W O
       |<────── 4N² + 4Nd ──────────────→|

FlashAttention:
  时间 ──────────────────────────────────→
  外循环 0:
       | Q₀K₀ | Q₀K₁ | Q₀K₂ | ... |
       |──────|──────|──────|     |  内循环
  HBM:  R Q₀ K₀ V₀   R K₁ V₁   R K₂ V₂
       |<── 只在内外循环之间读 Q/K/V ──→|
       中间结果 S, P 在 SMEM 中，不碰 HBM

                                        HBM 对比总结:
                                        ┌─────────────────────┐
                                        │ Standard: 4N² + 4Nd │
                                        │ Flash:    4Nd       │
                                        │ 节省:     N 倍!     │
                                        └─────────────────────┘
```

---

## 3. Block Size 怎么选

### 约束：Shared Memory 有限

```
Thor 每 SM 有 48 KB Shared Memory

FlashAttention 每块需要 SMEM:
  ┌──────────────────────────────────────┐
  │  Q_block: B_r × d × 2B  (FP16)      │
  │  K_block: B_c × d × 2B              │
  │  S_block: B_r × B_c × 2B ← 最大的!  │
  │  V_block: B_c × d × 2B              │
  │  O_block: B_r × d × 2B              │
  │                                      │
  │  总计 ≈ B_r×d + B_c×d + B_r×B_c     │
  │        + B_c×d + B_r×d  (× 2B)      │
  └──────────────────────────────────────┘

S_block 是瓶颈：B_r × B_c 必须 ≤ SMEM 的一半
```

### B_r, B_c 配置对比

```
                        SMEM 布局图 (B_r=B_c=64):
┌────────────────────────────────────────────────────────┐
│ Shared Memory (48 KB)                                  │
│                                                        │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Q_block: 8 KB    │  │ K_block: 8 KB    │           │
│  │  64 × 64 × 2B    │  │  64 × 64 × 2B    │           │
│  └──────────────────┘  └──────────────────┘           │
│                                                        │
│  ┌──────────────────────────────────┐  ┌────────────┐ │
│  │ S_block: 8 KB                     │  │ V_block:   │ │
│  │ 64 × 64 × 2B                      │  │ 8 KB       │ │
│  │  ← 每次内循环重算这张              │  │            │ │
│  └──────────────────────────────────┘  └────────────┘ │
│                                                        │
│  总计: 32 KB  ≤  48 KB  ✓                             │
└────────────────────────────────────────────────────────┘

                        B_r=B_c=128 时溢出!
┌────────────────────────────────────────────────────────┐
│ Shared Memory (48 KB)                                  │
│                                                        │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Q_block: 16 KB   │  │ K_block: 16 KB   │           │
│  │ 128 × 64 × 2B    │  │ 128 × 64 × 2B    │  ← 已经  │
│  └──────────────────┘  └──────────────────┘    32 KB  │
│                                                        │
│  ┌──────────────────────────────────┐                  │
│  │ S_block: 32 KB ←─ 超出 SMEM!     │                 │
│  │ 128 × 128 × 2B = 32 KB          │                  │
│  │ 加上 Q+K 已经 64 KB > 48 KB ✗   │                 │
│  └──────────────────────────────────┘                  │
└────────────────────────────────────────────────────────┘
```

### 分块循环的可视化

```
Q 矩阵 (N×d)          K 矩阵 (N×d)         V 矩阵 (N×d)         O 矩阵 (N×d)
┌──────────┐          ┌──────────┐         ┌──────────┐         ┌──────────┐
│ Q₀  ← B_r│          │ K₀ K₁ K₂ │         │ V₀ V₁ V₂ │         │  O₀      │
│ ──────── │          │ ↑        │         │ ↑        │         │          │
│ Q₁       │          │ B_c      │         │ B_c      │         │  O₁      │
│          │          │          │         │          │         │          │
│ Q₂       │          │          │         │          │         │  O₂      │
│          │          │          │         │          │         │          │
└──────────┘          └──────────┘         └──────────┘         └──────────┘
    │                     │                    │                    │
    └── 外层循环 ──► Q₀, Q₁, Q₂...            │                    │
                    │                         │                    │
                    ▼                         │                    │
              ┌───────────────────┐           │                    │
              │  内层循环:        │           │                    │
              │  for col=0,1,2:  │           │                    │
              │    load K_col     │◄──────────┘                    │
              │    load V_col     │◄───────────────┘               │
              │    S = Q_row × K  │                               │
              │    softmax merge  │                               │
              │    O_row += P×V   │───────────────────────────────►│
              └───────────────────┘                               │
                                                                  │
                Q_row 在内层循环中固定，复用计算                        │
                K/V 的每一列只被读取一次                               │
                                                                  │
  最终:                                                           │
    O₀ = softmax(Q₀K₀) @ V₀  +  softmax(Q₀K₁) @ V₁  +  ... ──────► O₀
```

---

## 4. PyTorch 简化实现

```python
import torch
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """Standard attention: 需要完整 N×N 矩阵。"""
    S = Q @ K.transpose(-2, -1)          # (N, N)  ← 写回 HBM
    P = F.softmax(S, dim=-1)             # (N, N)  ← 读 S, 写 P
    O = P @ V                            # (N, d)  ← 读 P, 写 O
    return O

def flash_attention(Q, K, V, B_r=64, B_c=64):
    """
    FlashAttention 简化版。
    和 Step 03 tiled matmul 完全一样的结构。
    """
    N, d = Q.shape
    O = torch.zeros(N, d, device='cuda', dtype=Q.dtype)

    # 外层循环：遍历 Q 的行
    for row_start in range(0, N, B_r):
        row_end = min(row_start + B_r, N)
        Q_block = Q[row_start:row_end]    # (B_r, d)

        # online softmax 的局部统计量
        m = torch.full((row_end - row_start, 1), -float('inf'), device='cuda')
        l = torch.zeros((row_end - row_start, 1), device='cuda')
        O_block = torch.zeros(row_end - row_start, d, device='cuda')

        # 内层循环：遍历 K/V 的行
        for col_start in range(0, N, B_c):
            col_end = min(col_start + B_c, N)
            K_block = K[col_start:col_end]  # (B_c, d)
            V_block = V[col_start:col_end]  # (B_c, d)

            # S_block = Q_block @ K_blockᵀ (B_r × B_c)
            S_block = Q_block @ K_block.T

            # online softmax
            block_m = S_block.max(dim=-1, keepdim=True).values
            block_p = torch.exp(S_block - block_m)
            block_l = block_p.sum(dim=-1, keepdim=True)

            # 合并统计量
            new_m = torch.maximum(m, block_m)
            l = torch.exp(m - new_m) * l + torch.exp(block_m - new_m) * block_l
            m = new_m

            # O_block += exp(S - m) @ V (需要 rescale)
            O_block = O_block * torch.exp(m - m)
            O_block += block_p @ V_block

        # 最终归一化
        O_block = O_block / l
        O[row_start:row_end] = O_block

    return O
```

---

## 5. Triton 实现

`demo_flash.py` 包含完整的 Triton FlashAttention forward kernel。

Triton 比 CUDA C++ 更适合实现 FlashAttention，因为：

```
CUDA C++:                          Triton:
───────                             ──────
手写 shared memory 管理             自动分配 SMEM
手写 __syncthreads()                大部分不需要同步
手写 warp-level 同步和 shuffle      自动处理
手动选寄存器变量                    编译器优化
需要处理 bank conflict              自动规避

→ FlashAttention 在 Triton 里    → 代码量少 3x
  需要大量 SMEM 和同步管理          更易调试
```

### Triton kernel 结构

```
                    ┌─────────────────────────────────┐
                    │  @triton.jit                    │
                    │  def flash_kernel(Q, K, V, O,  │
                    │          N, d, B_r, B_c):      │
                    │                                │
 每个 program       │    pid = tl.program_id(0)       │
 处理 Q 的          │    row_start = pid * B_r        │
 B_r 行             │                                │
                    │    # 加载 Q_block 到 SMEM       │
                    │    Q_block = tl.load(Q + ...)   │
                    │                                │
                    │    # 内层循环遍历 K/V           │
                    │    for col in range(N, B_c):    │
                    │      K_block = tl.load(K + ...) │
                    │      V_block = tl.load(V + ...) │
                    │      S = tl.dot(Q_block,        │
                    │                K_block.T)       │
                    │      # online softmax           │
                    │      block_m = tl.max(S, axis=1)│
                    │      block_p = tl.exp(S-bl_m)   │
                    │      block_l = tl.sum(block_p)  │
                    │      # merge + O accum         │
                    │      acc = acc * alpha         │
                    │         + beta * tl.dot(P,V_blk)│
                    │                                │
                    │    O_row = acc / l             │
                    │    tl.store(O + row_start, O)  │
                    └─────────────────────────────────┘
```

---

## 6. DSA: DeepSeek Sparse Attention — 从"怎么算"到"算什么"

### 6.1 DSA 和 FlashAttention 的互补关系

```
FlashAttention 解决的是：                DSA 解决的是：
   attention 的 HBM 访问问题                attention 的计算量问题

  Flash:                                  DSA:
  所有 Q 对所有 K（N×N）                   只选重要的 Q-K 对算
  但中间结果不落 HBM                       跳过的对根本不计算
  HBM 从 O(N²) → O(N)                    计算量从 O(N²) → O(N×k)

  类比：
  你在 Step 03 学 matmul 优化：
    naive: 每次从显存读，算完写回            ← HBM 问题
    tiled: 搬到 SMEM 里算，复用数据          ← FlashAttention 的思路

  DSA 是另一个维度：
    有些 Q-K 对的 attention score 接近 0
    意味着它们对最终输出几乎没有贡献
    那为什么还要算它们？
    → DSA 跳过它们

  FlashAttention + DSA = 互补：
    FlashAttention: 对"必须算的"算得高效
    DSA:           决定"哪些可以不算"
```

### 6.2 DSA 的核心：Gating Network

```
DSA 靠一个轻量的 gating network 决定每个 Q 应该看哪些 K。

标准 attention:
  for each Q:
      scores = Q @ Kᵀ          # (1 × d) × (d × N) → (1 × N)
      probs = softmax(scores)  # 全部 N 个 K 都参与

DSA:
  for each Q:
      mask = gating_network(Q)  # (1 × N) 的 0/1 mask
      k_idx = where(mask == 1)  # 选出 k 个重要的 K
      scores = Q @ K[k_idx]ᵀ   # (1 × d) × (d × k) → (1 × k)
      probs = softmax(scores)  # 只在这 k 个上算

      # k << N, 比如 N=128K, k=16K
```

### Gating Network 的结构

```
Gating network 是一个小型的两层 MLP：

  Q_row (1 × d)
     │
     ▼
  ┌────────────────────────────────┐
  │  Linear(d → d_gate)            │  d_gate 很小，比如 64
  │  → ReLU                        │
  │  → Linear(d_gate → N)          │  输出 N 个 logits
  └────────────────────────────────┘
     │
     ▼
  top_k(logits, k)                取分数最高的 k 个 K 的位置
     │                               k = N × sparsity_ratio
     ▼
  indices (k,)                    要计算的 K 的索引列表

  开销：
    这个 gating network 的计算量 ≈ O(N × d_gate)
    比完整的 QKᵀ (O(N × d)) 少得多
    d_gate << d（比如 d=4096, d_gate=64）
```

### 6.3 DSA Kernel 的执行流程

```
                                         输入 Q, K, V
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Gating Network  │
                                    │ (小 MLP per Q)  │
                                    │ 输出 top-k 索引  │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Gather K, V     │
                                    │ 只取需要的行     │
                                    │ K_selected (k×d)│
                                    │ V_selected (k×d)│
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Sparse GEMM     │
                                    │ Q × K_selectedᵀ  │
                                    │ → S (1 × k)     │
                                    │                  │
                                    │ online softmax   │
                                    │ → P (1 × k)     │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Scatter Add     │
                                    │ P × V_selected   │
                                    │ → O (1 × d)     │
                                    │ 写回对应位置     │
                                    └─────────────────┘
```

### CPU vs GPU 上的执行

```
在 GPU 上实现 DSA 最关键的问题是"非连续访存"：

传统 dense GEMM:
  Q @ Kᵀ:
    读 Q: 连续 (d 个 float)
    读 K: 连续 (N × d 个 float)
    写 S: 连续 (N 个 float)
    → 完美 coalescing ✓

DSA 的 gather:
  Q @ K[top-k]ᵀ:
    读 Q: 连续 ✓
    读 K_selected: 非连续！
      每次随机挑一个 K_row，它在显存里可能在任何位置
      → 每次 gather 都是一次 random access
      → L2 cache miss 率很高
    写 S: 连续 (k 个 float) ✓

  DSA kernel 的优化重点：
    1. 把 gather 到的 K_row 连续写到一个临时缓冲区
    2. 然后再做连续的 GEMM
    3. 用 warp-level 协作 gather 减少 bank conflict
```

### 6.4 DSA 在 GPU 上的 Kernel 实现

```
DSA 的 GPU kernel 分两步（或 fused 成一步）：

Step 1: Gating + Gather Kernel

  ┌────────────────────────────────────────────────────┐
  │ grid: N 个 block（每个 Q 一行一个 block）           │
  │                                                    │
  │ block 做的事：                                     │
  │   Q_row = Q[pid]             # 从 HBM 加载         │
  │   logits = gate_mlp(Q_row)   # 小 MLP，在寄存器算  │
  │   topk_idx = topk(logits, k) # 找出 k 个最重要的 K │
  │                                                    │
  │   # 把这些 K 行连续写到缓冲区                        │
  │   for i in range(k):                               │
  │     K_buf[i] = K[topk_idx[i]]  # gather（非连续读） │
  │     V_buf[i] = V[topk_idx[i]]                      │
  └────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Sparse Attention Kernel (FlashAttention-like)

  ┌────────────────────────────────────────────────────┐
  │ grid: N 个 block                                   │
  │                                                    │
  │ block 做的事：                                     │
  │   Q_row = Q[pid]                                   │
  │                                                    │
  │   # 在 K_buf, V_buf 上做 FlashAttention 的 tiling  │
  │   for col in range(0, k, B_c):                     │
  │     K_block = K_buf[col:col+B_c]     # 连续读 ✓    │
  │     V_block = V_buf[col:col+B_c]                   │
  │     S = Q_row × K_blockᵀ                           │
  │     online_softmax(S)                              │
  │     O += P × V_block                               │
  │                                                    │
  │   # 把结果写回 O[pid]                               │
  └────────────────────────────────────────────────────┘
```

### 6.5 DSA 的 Tiling 策略——和 Step 03 一样

```
DSA kernel 的 tiling 和你在 Step 03 学的 tiled matmul
以及上面 FlashAttention 的 tiling 是一脉相承的：

                             ┌──────────────────────┐
  Step 03 tiled matmul:      │ for tk: C += A×B     │
                             │ A_block, B_block tiled│
                             └──────────┬───────────┘
                                        │
                             ┌──────────┴───────────┐
  FlashAttention tiling:     │ for col: O += P×V    │
                             │ Q_block, K_block tiled│
                             └──────────┬───────────┘
                                        │
                             ┌──────────┴───────────┐
  DSA tiling:                │ 先 gather: K_buf      │
                             │ 再 for col: O += P×V  │
                             │ 增加一步：非连续→连续  │
                             └──────────────────────┘

  区别：
    → Step 03:    tiling 是为了减少 DRAM 访问（复用 SMEM）
    → FlashAttn:  tiling 是为了不写 N×N 中间矩阵
    → DSA:        tiling + gather 是为了跳过不需要的 K
```

### 6.6 训练流程：两阶段

```
DSA 不是在推理时才"选"哪些 K 重要，
而是在训练时就学会哪些 K 不重要。

                   时间 ──────────────────────────→
                        │                          │
     dense warm-up      │      sparse training     │
     ┌──────────────────┼──────────────────────────┐
     │ 标准 attention    │ 引入 gating network       │
     │ 所有 Q-K 都算   │ 逐步增加稀疏度            │
     │ 训练 gate 的初值 │ λ 从 0 逐渐增大            │
     └──────────────────┼──────────────────────────┘
                        │                          │
     模型学语义表示      │ 模型学 attention 稀疏模式  │
     门控还没起作用      │ gate 学会"哪些 attention  │
                        │ 路径可以省"               │
     ┌──────────────────┼──────────────────────────┐
     │ loss             │ loss + λ × sparsity_loss │
     └──────────────────┴──────────────────────────┘

  两阶段的必要性：
    如果一开始就稀疏，模型会"走捷径"
    → 只关注局部模式，学不到全局依赖
    → 和人的学习过程一样：先广泛接触，再精准修剪

  稀疏度控制：
    sparsity = 1 - k / N
    V3.2 的 DSA sparsity ≈ 75-90%（看具体的层和 head）
    即每个 Q 只看 10-25% 的 K
```

### 6.7 DSA 的实现工具：TileLang

```
DeepSeek V3.2 的 DSA kernel 是用 TileLang 写的。
你之前问"TileLang 是什么"，这里就是答案：

  前几章的进度：
    Step 03: 手写 CUDA C++ tiled matmul
    Step 06: 用 Triton 写 GPU kernel
    ADVANCED/03: 用 Triton 写 FlashAttention

  DSA kernel 为什么要用 TileLang？
    ┌──────────────────────────────────────────────┐
    │ DSA kernel 的复杂性：                          │
    │   1. gather 阶段：非连续访存 → 需要手动优化    │
    │   2. gating MLP：小矩阵运算 → 要 fuse 进去    │
    │   3. sparse GEMM：变长 k → 动态 block 大小    │
    │   4. scatter 写回：非连续写 → 要处理 bank     │
    │                                               │
    │ 这些在 CUDA 里写 → 几千行，极易出错             │
    │ 在 Triton 里写 → tl.dot 期望连续张量，gather   │
    │                  操作不好表达                   │
    │ 在 TileLang 里写 → 声明式："我要对 attention   │
    │   矩阵做 top-k 稀疏"，编译器自动生成 CUDA      │
    └──────────────────────────────────────────────┘

  TileLang 和 Triton 的对比：
    Triton:  "怎么算"（你写每个 block 的代码）
    TileLang: "算什么"（你声明计算规则，编译器生成）

  三者的关系：
    CUDA C++:  手写所有细节（Step 03）
         ↓ 抽象
    Triton:    自动 SMEM / coalescing（Step 06）
         ↓ 抽象
    TileLang:  自动 tiling + sparse pattern（DSA）

  FlashMLA（DeepSeek 开源的 CUDA kernel）:
    实测 640 TFLOPS（预填充）, 410 TFLOPS（解码）
    是 DSA 的具体 CUDA 实现层
    TileLang → 编译器生成 → FlashMLA kernel
```

### 6.8 DSA + FlashAttention + MLA 的协同

```
                        │
             输入序列 N=128K
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │ MLA: 将 K, V 压缩为潜变量                      │
    │ KV cache: N×d → N×(d/4)                      │
    │ 显存省 4x                                     │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────┐
    │ DSA: gating network 选出每个 Q 要看的 K       │
    │ N→k, k≈N/4（稀疏度 75%）                      │
    │ 计算量省 4x                                   │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────┐
    │ FlashAttention: 对选出的 k 个 K 做在线 softmax│
    │ HBM 访问: O(N×k) → O(N)（因为 tiling）        │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
                     输出 O

  三者叠加的效果：
    MLA:  KV cache 省 4x  → 能装更长的上下文
    DSA:  计算量省 4x     → 更快的推理
    Flash: HBM 访问省 N/k  → 更高的带宽利用率

  这就是 DeepSeek 128K 上下文背后的技术栈
```

### 6.9 和本课程的直接连接

```
你在本课程学到的每个概念，都在 DSA 里用上了：

  Step 01 带宽:   DSA 的 gather 需要高带宽（非连续访存）
  Step 02 roofline: DSA 把 AI 从 d/2 提升到 N/utils
  Step 03 tiling:  DSA = gather + tiled attention
  Step 04 nsys:    DSA kernel 在 nsys 上可以看到
                   稀疏 attention 的 gather 阶段
  Step 05 bottleneck: DSA 解决的就是 attention 的
                      compute-bound 瓶颈
  Step 06 Triton:  理解 Triton 才能理解为什么 DSA
                   需要 TileLang（Triton 不够灵活）
  ADV/01 CUDA Graph: DSA 的 gating + gather 可以
                     graph capture
  ADV/02 compile:   TileLang ≈ 编译器的 torch.compile
                    但针对 attention 做了特殊优化

  DSA = 整个课程知识点的汇合
```

---

## 7. 性能预期

### Standard 的 N² 增长 vs Flash 的 Nd 增长

```
时间 (ms)
  │
  │
4.0 │                          ● Standard Attention
  │                           (≈ 0.08 × N² / 1024²)
3.0 │
  │
2.0 │
  │                     ●
1.0 │
  │              ●
  │       ●
0.0 └───●───●───●───────────────────────────→ N
      256  512 1024 2048  4096  8192
                      │
                      │  ▲ FlashAttention
                      │  (≈ 0.04 × N / 1024)
                      │  几乎是一条平线

N     Standard (approx)   FlashAttention   加速比
─────────────────────────────────────────────────
256       0.005 ms          0.005 ms         ~1x
512       0.020             0.010             2x
1024      0.080             0.020             4x
2048      0.320             0.040             8x
4096      1.280             0.080            16x
8192      5.120             0.160            32x
```

### Roofline 上 Standard 和 Flash 的位置

```
                                GFLOP/s ↑
                                        │
                                  算力天花板 ──────────────────● FlashAttention
                                        │                      AI ≈ 4N²d / 4Nd
                                        │                           = N
                                        │
                                        │
                                        │            ● Standard
                                        │            AI = d/2
                                  带宽天花板 ──╲
                                        │     ╲
                                        │      ╲
                                        └──────────────────────→ FLOP/Byte
                                        ridge
                                        point
                                        (AI=~34)

  Standard:   AI = d/2 = 32        → ridge_point 附近
  FlashAttention: AI = N           → N > 64 时就是 compute-bound!
                                      在 roofline 上往右移动了很多

  → Standard 受带宽限制
  → Flash 受算力限制（N 越大越明显）
```

---

## 8. 和前面步骤的连接

```
Step 02 roofline:
  Standard attention:          AI = d/2
  FlashAttention:              AI = N
  DSA:                        AI = N / sparsity_ratio
  → N 越大，Flash 在 roofline 上越靠右
  → DSA 再往左推（因为更少的 FLOP 被浪费）

Step 03 tiled matmul:
  FlashAttention 的结构 = Step 03 tiled matmul
  + online softmax 作为累加器
  DSA = FlashAttention + gating network

  tiled matmul:   for tk:    C += A_block × B_block
  FlashAttention:  for col:   O += softmax(Q×K_block) × V_block
  DSA:             for col:   O += softmax(Q×K_gathered[col]) × V_gathered[col]

Step 06 Triton:
  demo_flash.py 的 Triton kernel
  tl.dot 映射到 Tensor Core
  DSA 的 gating 部分超出 Triton 的能力范围
  → 需要 TileLang 或手写 CUDA

ADVANCED 系列:
  FlashAttention = "把必须算的算好"
  DSA = "决定哪些可以不算是"  → 更高层的抽象
  torch.compile = 自动融合
  CUDA Graph = 消除启动开销
```

---

## 9. 总结

```

Standard Attention:          FlashAttention:              DSA:
  ┌─────┐┌─────┐┌─────┐     ┌──────────────────┐        ┌────────────────────┐
  │QKᵀ  ││Soft ││PV   │     │ for each block:  │        │ gate: 选出重要 K   │
  │     ││     ││     │     │ S=Q_blk×K_blk    │        │ gather: K_buf      │
  │N×N  ││N×N  ││N×N  │     │ online_softmax   │        │ FlashAttention     │
  │WRITE││READ ││READ │     │ O_blk+=P×V_blk   │        │ on gathered K_buf  │
  └─────┘└─────┘└─────┘     │ 中间不碰 HBM     │        └────────────────────┘
  HBM: O(N²)                HBM: O(N)                   计算: O(N×k), k<<N

三行代码理解:
  原来: S=Q@Kᵀ → P=softmax(S) → O=P@V
  Flash: for 块: S=Q_block@K_blockᵀ → online_softmax → O+=P@V_block
  DSA:   idx=gate(Q); K_buf=K[idx]; FlashAttention(Q, K_buf, V_buf)

FlashAttention 和 DSA 的关系:
  FlashAttention 是"怎么算"—— tiling + online softmax
  DSA 是"算什么"—— 哪些 Q-K 对值得算
  两者互补，都用在 DeepSeek V3.2 的生产推理中

和本课程的关系:
  FlashAttention = Step 03 tiling + Step 02 roofline
  DSA = Step 05 bottleneck + gating network
  TileLang = 比 Triton 更高层的"声明式计算"
  整个 ADVANCED 系列 = 把 Steps 01-06 的知识点推到前沿论文
```
