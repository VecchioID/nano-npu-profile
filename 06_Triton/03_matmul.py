"""
Step 03 Triton 版：Matmul 三层优化
===================================
对应 C++ 版 Step 03：naive → tiled → Tensor Core（tl.dot）

Triton 里没有"naive kernel"——因为 Triton 本身就是 tile 编程模型。
但我们可以模拟三种层次的写法来对比理解。

运行：python3 03_matmul.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════
# 第一层：模拟「naive」—— 每个 block 处理 1 个元素
# ═══════════════════════════════════════════════════════════════
# 相当于 CUDA 的 thread-per-element naive matmul
# 在 Triton 里故意把 BLOCK 设成 1 来模拟

@triton.jit
def naive_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    """每个 program 处理 1 个输出元素。故意低效。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = 0.0
    for k in range(K):
        a = tl.load(a_ptr + pid_m * stride_am + k * stride_ak)
        b = tl.load(b_ptr + k * stride_bk + pid_n * stride_bn)
        acc += a * b

    tl.store(c_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


# ═══════════════════════════════════════════════════════════════
# 第二层：tile 版（手动 shared memory）
# ═══════════════════════════════════════════════════════════════
# 和 Step03 的 tiled_matmul 对应
# Triton 自动管理的 shared memory，但我们显式写出 tiling 逻辑

@triton.jit
def tiled_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Tiled matmul：每个 block 处理 BM×BN tile，K 方向每次读 BK。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N)
        acc += tl.dot(a, b)   # ← 这里用了 Tensor Core！
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# ═══════════════════════════════════════════════════════════════
# 第三层：完全体 matmul（group scheduling + 全 tile）
# ═══════════════════════════════════════════════════════════════
# Triton 官方示例风格，性能和 cuBLAS 接近

@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GM': 8}, num_warps=4),
        triton.Config({'BM': 256, 'BN': 64,  'BK': 32, 'GM': 8}, num_warps=4),
        triton.Config({'BM': 64,  'BN': 256, 'BK': 32, 'GM': 8}, num_warps=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GM': 8}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fast_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr,
):
    """带 group scheduling 的 TL matmul。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    group_size_m = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def bench_matmul(fn, M, N, K, dtype=torch.float16, warmup=5, runs=50):
    a = torch.randn(M, K, device='cuda', dtype=dtype)
    b = torch.randn(K, N, device='cuda', dtype=dtype)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)

    if fn == naive_matmul:
        grid = lambda _: (M, N)
        kwargs = {}
    elif hasattr(fn, 'cache_key'):  # autotuned — no explicit BM/BN/BK
        grid = lambda meta: (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
        kwargs = {}
    else:
        grid = lambda meta: (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
        kwargs = {'BM': 128, 'BN': 128, 'BK': 32}

    for _ in range(warmup):
        fn[grid](a, b, c, M, N, K, a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1), c.stride(0), c.stride(1), **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn[grid](a, b, c, M, N, K, a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1), c.stride(0), c.stride(1), **kwargs)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / runs
    flops = 2 * M * N * K
    gflops = flops / ms / 1e6
    bytes_ = (M * K + K * N) * a.element_size() + M * N * c.element_size()
    gbps = bytes_ / ms / 1e6
    ai = flops / bytes_
    return ms, gflops, gbps, ai, c


def main():
    print("=" * 60)
    print("Step 03 Triton 版：Matmul 三层优化")
    print("=" * 60)
    print()
    print("Triton 本身就是 tile 编程模型。这里的「三层」对应：")
    print("  第一层 (naive):  BLOCK=1，每个 program 算 1 个元素")
    print("  第二层 (tiled):  BLOCK=128，用 tl.dot (Tensor Core)")
    print("  第三层 (fast):   带 autotune + group scheduling")
    print()

    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 1024)]
    kernels = [
        ("naive (BLOCK=1, FP32)", naive_matmul, torch.float32),
        ("tiled (tl.dot FP16)", tiled_matmul, torch.float16),
        ("fast (autotune FP16)", fast_matmul, torch.float16),
    ]

    for label, fn, dtype in kernels:
        print(f"\n── {label} ──")
        for M, N, K in sizes:
            try:
                ms, gflops, gbps, ai, _ = bench_matmul(fn, M, N, K, dtype, runs=5 if fn == naive_matmul else 50)
                print(f"  {M:4d}³: {ms:8.3f} ms, {gflops:8.0f} GFLOP/s, AI={ai:.1f}")
            except Exception as e:
                print(f"  {M:4d}³: error — {e}")

    # cuBLAS 对比
    print("\n── cuBLAS (torch.matmul FP16) ──")
    for M, N, K in sizes:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 100
        flops = 2 * M * N * K
        gflops = flops / ms / 1e6
        print(f"  {M:4d}³: {ms:8.3f} ms, {gflops:8.0f} GFLOP/s")

    print()
    print(f"结果对比 C++ 版 (Step 03 WMMA 1024³ ≈ 3,955 GFLOP/s):")
    print(f"  如果 Triton fast_matmul 接近这个值，说明 Triton 不输手写 WMMA")
    print("=" * 60)


if __name__ == "__main__":
    main()
