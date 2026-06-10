"""
Step 05 Triton 版：Bottleneck 分类分析
=======================================
对应 C++ 版 bound_analysis.cu + fix_perf.cu

运行：python3 05_bottleneck.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════
# 1. Memory-bound：简单 copy
# ═══════════════════════════════════════════════════════════════

@triton.jit
def mem_bound_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


# ═══════════════════════════════════════════════════════════════
# 2. Compute-bound：大量数学运算
# ═══════════════════════════════════════════════════════════════

@triton.jit
def compute_bound_kernel(data_ptr, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(data_ptr + offsets, mask=mask)
    for _ in range(ITERS):
        x = tl.sin(tl.cos(x * x + 0.5))
        x = tl.exp(tl.log(tl.abs(x) + 1e-6))
    tl.store(data_ptr + offsets, x, mask=mask)


# ═══════════════════════════════════════════════════════════════
# 3. 平衡型：适量计算
# ═══════════════════════════════════════════════════════════════

@triton.jit
def balanced_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    for _ in range(8):
        x = x * 0.5 + 0.5
    tl.store(out_ptr + offsets, x, mask=mask)


# ═══════════════════════════════════════════════════════════════
# 4. 非合并访问（用不同的 stride 模拟）
# ═══════════════════════════════════════════════════════════════

@triton.jit
def strided_kernel(in_ptr, out_ptr, n, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # 按 stride 跳跃式读取
    x = tl.load(in_ptr + offsets * STRIDE, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


# ═══════════════════════════════════════════════════════════════
# 5. 低占用率（用不同的 BLOCK 大小模拟）
# ═══════════════════════════════════════════════════════════════

@triton.jit
def simple_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + 1.0, mask=mask)


# ═══════════════════════════════════════════════════════════════
# Reduction 三种实现
# ═══════════════════════════════════════════════════════════════

@triton.jit
def reduce_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Triton 版规约 — 用 tl.sum 自动树形规约。"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(x)
    tl.store(out_ptr + pid, acc)


def format_gbps(bytes, ms):
    return bytes / ms / 1e6


def format_tflops(flops, ms):
    return flops / ms / 1e12


def main():
    print("=" * 60)
    print("Step 05 Triton 版：Bottleneck 分类分析")
    print("=" * 60)

    N = 8 * 1024 * 1024
    bytes_per = N * 4
    device = torch.cuda.current_device()

    grid_fn = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    x = torch.randn(N, device='cuda')
    y = torch.empty(N, device='cuda')

    def bench(fn, grid, args, warmup=5, runs=100):
        if grid is None:
            grid = grid_fn
        for _ in range(warmup):
            fn[grid](*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            fn[grid](*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / runs

    # ── 1. Memory-bound ──
    print("\n--- 1. Memory-Bound ---")
    ms_mem = bench(mem_bound_kernel, None, (x, y, N, 1024))
    bw = format_gbps(bytes_per * 2, ms_mem)
    print(f"  mem-bound (1 read + 1 write): {ms_mem:.3f} ms, {bw:.0f} GB/s")

    # ── 2. Compute-bound ──
    print("\n--- 2. Compute-Bound ---")
    ms100 = bench(compute_bound_kernel, None, (y, N, 100, 1024), warmup=2, runs=10)
    ms1000 = bench(compute_bound_kernel, None, (y, N, 1000, 1024), warmup=2, runs=5)
    print(f"  compute-bound (100 iters):  {ms100:.3f} ms")
    print(f"  compute-bound (1000 iters): {ms1000:.3f} ms")
    print(f"  线性增长? {ms1000/ms100:.1f}x (期望 10x)")

    # ── 3. Balanced ──
    print("\n--- 3. Balanced ---")
    ms = bench(balanced_kernel, None, (x, y, N, 1024))
    bw = format_gbps(bytes_per * 2, ms)
    print(f"  balanced (8 flops/elem): {ms:.3f} ms, {bw:.0f} GB/s")

    # ── 4. Strided access ──
    print("\n--- 4. Strided Access ---")
    big = torch.randn(N * 32, device='cuda')
    stride_times = {}
    for stride, label in [(1, "stride=1 (coalesced)"),
                           (4, "stride=4 (bad)"),
                           (32, "stride=32 (worst)")]:
        ms = bench(strided_kernel, None, (big, y, N, stride, 1024))
        stride_times[stride] = ms
        print(f"  {label}: {ms:.3f} ms")
    ms_stride1 = stride_times[1]
    ms_stride32 = stride_times[32]

    # ── 5. Occupancy ──
    print("\n--- 5. Occupancy ---")
    g32 = lambda meta: (triton.cdiv(N, 32),)
    ms_low = bench(simple_kernel, g32, (x, y, N, 32))
    ms_high = bench(simple_kernel, None, (x, y, N, 1024))
    print(f"  low occ (32 thr):  {ms_low:.3f} ms")
    print(f"  high occ (1024):   {ms_high:.3f} ms")
    print(f"  加速比 (32→1024): {ms_low/ms_high:.2f}x")

    # ── 6. Reduction ──
    print("\n--- 6. Reduction ---")
    N_red = 64 * 1024 * 1024
    grid_red = lambda meta: (triton.cdiv(N_red, meta['BLOCK']),)
    r = torch.randn(N_red, device='cuda')
    o = torch.empty(triton.cdiv(N_red, 1024), device='cuda')

    ms_red = bench(reduce_kernel, grid_red, (r, o, N_red, 1024), runs=100)
    print(f"  tl.sum reduction: {ms_red:.3f} ms")
    print(f"  (Triton tl.sum 自动做树形规约，无需手写三种版本对比)")

    print("\n" + "=" * 60)
    print("与 C++ 版 bound_analysis 结果对比：")
    print("  C++ mem-bound:     0.707 ms")
    print(f"  Triton mem-bound:  {ms_mem:.3f} ms")
    print("  C++ stride=32:     6.578 ms (8.0x vs stride=1)")
    print(f"  Triton stride=32:  {ms_stride32:.3f} ms ({ms_stride32/ms_stride1:.1f}x vs stride=1)")
    print("  C++ low occ:       1.598 ms (2.3x vs 256)")
    print(f"  Triton low occ:    {ms_low:.3f} ms ({ms_low/ms_high:.2f}x vs 1024)")
    print("=" * 60)


if __name__ == "__main__":
    main()
