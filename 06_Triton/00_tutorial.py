"""
Triton 入门教程
==============

Triton = Python 写的 GPU kernel 编译器。
你描述"每个 block 做什么"，Triton 自动帮你生成 CUDA 代码。

对比 CUDA C++：
  CUDA:  你写线程（threadIdx.x），手动管 shared memory、warp、coalescing
  Triton:你写 block（tl.program_id），自动管 shared memory、warp、coalescing

三步写一个 Triton kernel：
  1. @triton.jit 装饰一个函数
  2. 用 tl.program_id() 拿到 block 编号
  3. 用 tl.load / tl.store 访存，用 tl.dot / tl.sum 等做计算

运行：python3 00_tutorial.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════
# 0. 第一个 kernel：向量加法
# ═══════════════════════════════════════════════════════════════

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    每个 block 处理 BLOCK 个元素。

    CUDA 版等价代码：
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = x[idx] + y[idx];
    """
    pid = tl.program_id(0)                  # 相当于 blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 相当于 threadIdx.x 的向量版本
    mask = offsets < n                      # 边界检查
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# ═══════════════════════════════════════════════════════════════
# 1. 核心概念
# ═══════════════════════════════════════════════════════════════

"""
1.1  tl.program_id(axis)  →  blockIdx.x/y/z
     每个 block 有唯一的 program_id。axis=0/1/2 对应 x/y/z。

1.2  tl.arange(0, BLOCK)  →  连续整数向量
     返回 [0, 1, 2, ..., BLOCK-1]。加上 pid*BLOCK 得到全局偏移。

1.3  tl.constexpr          →  编译时常量
     BLOCK 在编译时确定，Triton 根据它做循环展开、向量化。

1.4  tl.load / tl.store    →  访存
     自动处理向量化、合并访问、边界检查（mask 参数）。

1.5  mask                  →  边界保护
     当 n 不能被 BLOCK 整除时，最后一个 block 的部分线程不应访存。

1.6  你写 block，不是线程
     tl.arange(BLOCK) 返回一个包含 BLOCK 个元素的向量。
     Triton 自动把它映射到 warp 上。你不需要知道 warp 是什么。
"""


# ═══════════════════════════════════════════════════════════════
# 2. tl.arange 详解
# ═══════════════════════════════════════════════════════════════

@triton.jit
def debug_arange_kernel(out_ptr, BLOCK: tl.constexpr):
    """把当前 program_id 和 arange 值写入输出，方便理解。"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    # 写入每个位置自己的编号
    tl.store(out_ptr + offsets, offsets)


def demo_arange():
    """演示 program_id 和 arange 如何映射到线程。"""
    BLOCK = 8
    out = torch.empty(32, device='cuda', dtype=torch.float32)
    grid = (4,)  # 4 个 block
    debug_arange_kernel[grid](out, BLOCK=BLOCK)
    print("tl.arange 演示（4 blocks × 8 elements）：")
    print("  output:", out.cpu().tolist())
    print("  等价 CUDA: threadIdx.x + blockIdx.x * blockDim.x")
    print()


# ═══════════════════════════════════════════════════════════════
# 3. 二维 grid：矩阵逐元素操作
# ═══════════════════════════════════════════════════════════════

@triton.jit
def elementwise_2d(x_ptr, out_ptr, M, N,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """二维 grid，每个 block 处理一个 BLOCK_M × BLOCK_N 的 tile。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], x * 2.0, mask=mask)


def demo_2d():
    M, N = 32, 64
    x = torch.randn(M, N, device='cuda')
    out = torch.empty(M, N, device='cuda')
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))
    elementwise_2d[grid](x, out, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = x * 2.0
    print("2D elementwise:", (out - ref).abs().max().item())
    print()


# ═══════════════════════════════════════════════════════════════
# 4. tl.dot：Tensor Core 矩阵乘
# ═══════════════════════════════════════════════════════════════

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """标准 Triton matmul kernel。一行 tl.dot 调用 Tensor Core。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def demo_matmul():
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    ref = torch.matmul(a.float(), b.float())
    print("Triton matmul @ 1024³ C++:", (c - ref).abs().max().item())
    print()


# ═══════════════════════════════════════════════════════════════
# 5. Autotune：自动搜索最优参数
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotune(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """带 autotune 和 group scheduling 的 matmul。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# ═══════════════════════════════════════════════════════════════
# 6. 常用操作速查
# ═══════════════════════════════════════════════════════════════

"""
访存：
  tl.load(ptr, mask=mask, other=0.0)     # 读，越界时返回 other
  tl.store(ptr, val, mask=mask)           # 写
  ptr += stride                           # 指针算术（注意 tensor 指针的偏移）

数学：
  tl.dot(a, b)                            # Tensor Core 矩阵乘（FP16/BF16/INT8）
  tl.sum(x, axis=0/1)                     # 规约求和
  tl.max(x, axis=0/1)                     # 规约求最大
  tl.exp(x), tl.log(x), tl.sin(x)         # 数学函数
  tl.abs(x), tl.sqrt(x), tl.where(c, x, y) # 条件/绝对值/平方根
  tl.fdiv(x, y)                           # 浮点除法

类型：
  tl.float16, tl.bfloat16, tl.float32     # 浮点
  tl.int8, tl.int16, tl.int32, tl.int64   # 整数
  tl.uint8, tl.uint16, tl.uint32          # 无符号

其他：
  tl.arange(0, N)                         # [0, 1, ..., N-1]
  tl.cdiv(a, b)                           # ceil(a/b)
  tl.max_contiguous(arange, N)            # 确保 arange 在 N 的边界内连续
  tl.zeros/ones(shape, dtype)             # 初始化
"""


# ═══════════════════════════════════════════════════════════════
# 7. CUDA ↔ Triton 对照表
# ═══════════════════════════════════════════════════════════════

"""
概念              CUDA C++                      Triton
─────             ───────                       ──────
block id          blockIdx.x                    tl.program_id(0)
thread id         threadIdx.x                   tl.arange(0, BLOCK)
全局偏移          blockIdx.x*blockDim.x          pid*BLOCK + tl.arange(0, BLOCK)
                         + threadIdx.x
grid 维度         dim3 grid(B, G)                grid = (B, G)
block 维度        dim3 block(T)                  BLOCK: tl.constexpr
边界检查          if (idx < n)                   mask = offsets < n
shared memory     __shared__ float s[N];         tl.zeros([N], dtype)（在 kernel 内定义）
同步              __syncthreads()                tl.debug_barrier()（通常不需要）
Tensor Core       wmma::fragment / mma_sync      tl.dot(a, b)
指针算术          ptr + idx                      （Triton 自动处理）
编译期常量        #define / constexpr            tl.constexpr
Autotuning        手写多版本                     @triton.autotune(configs=[...])
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Triton 入门教程")
    print("=" * 60)
    print()

    # 测试向量加法
    n = 1024 * 1024
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    out = torch.empty(n, device='cuda')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    ref = x + y
    print("向量加法:", "通过" if (out - ref).abs().max().item() < 1e-5 else "失败")

    demo_arange()
    demo_2d()
    demo_matmul()

    print("=" * 60)
    print("教程完成！继续运行 01_baseline.py")
    print("=" * 60)
