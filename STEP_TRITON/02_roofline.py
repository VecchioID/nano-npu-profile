"""
Step 02 Triton 版：Roofline 数据生成
====================================
运行：python3 02_roofline.py

生成 roofline_triton.png，与 C++ 版 roofline.png 对比。
"""

import torch
import triton
import triton.language as tl
import subprocess, sys, math


# ═══════════════════════════════════════════════════════════════
# 各种 kernel
# ═══════════════════════════════════════════════════════════════

@triton.jit
def copy_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def saxpy_kernel(x_ptr, y_ptr, out_ptr, n, a: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a * x + y, mask=mask)


@triton.jit
def fma_kernel(data_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(data_ptr + offsets, mask=mask)
    for _ in range(100):
        x = x * 1.05 + 0.01
    tl.store(data_ptr + offsets, x, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
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
        acc += tl.dot(a, b)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def bench(fn, grid, *args, warmup=3, runs=50, **kwargs):
    """运行 kernel fn，返回 (avg_ms, flops, bytes)。"""
    for _ in range(warmup):
        fn[grid](*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn[grid](*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def main():
    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    print("=" * 60)
    print("Step 02 Triton 版：Roofline 数据生成")
    print("=" * 60)

    N = 16 * 1024 * 1024  # 16M elements
    results = []

    # ── copy ──
    x = torch.randn(N, device='cuda')
    y = torch.empty(N, device='cuda')
    g = lambda m: (triton.cdiv(N, m['BLOCK']),)
    ms = bench(copy_kernel, g, x, y, N, BLOCK=1024, runs=100)
    flops = 0
    bytes = N * 4 * 2  # read + write
    ai = 0
    gbps = bytes / ms / 1e6
    results.append(('copy', ms, flops, bytes, ai, gbps))
    print(f"  copy: {ms:.3f} ms, {gbps:.0f} GB/s")

    # ── saxpy ──
    ms = bench(saxpy_kernel, g, x, y, y, N, a=2.0, BLOCK=1024, runs=100)
    flops = N * 2  # mul + add
    bytes = N * 4 * 3  # 2 read + 1 write
    ai = flops / bytes
    gbps = bytes / ms / 1e6
    gflops = flops / ms / 1e6
    results.append(('saxpy', ms, flops, bytes, ai, gbps))
    print(f"  saxpy: {ms:.3f} ms, {gflops:.0f} GFLOP/s, AI={ai:.1f}")

    # ── fma (compute-bound) ──
    ms = bench(fma_kernel, g, x, N, BLOCK=1024, runs=20)
    flops = N * 2 * 100  # 100 iters × (mul+add)
    bytes = N * 4 * 2  # read + write
    ai = flops / bytes
    gflops = flops / ms / 1e6
    results.append(('fma_x100', ms, flops, bytes, ai, 0))
    print(f"  fma_x100: {ms:.3f} ms, {gflops:.0f} GFLOP/s, AI={ai:.1f}")

    # ── matmul @ 1024³ ──
    M = N = K = 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    g2 = lambda m: (triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
    ms = bench(matmul_kernel, g2,
               a, b, c, M, N, K,
               a.stride(0), a.stride(1),
               b.stride(0), b.stride(1),
               c.stride(0), c.stride(1),
               BM=128, BN=128, BK=32, runs=10)
    flops = 2 * M * N * K
    bytes = (M * K + K * N) * 2 + M * N * 4  # A+B(fp16) + C(fp32)
    ai = flops / bytes
    gflops = flops / ms / 1e6
    results.append(('matmul_1024', ms, flops, bytes, ai, 0))
    print(f"  matmul 1024³: {ms:.3f} ms, {gflops:.0f} GFLOP/s, AI={ai:.1f}")

    # ── matmul @ 4096 ──
    M = N = K = 4096
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    ms = bench(matmul_kernel, g2,
               a, b, c, M, N, K,
               a.stride(0), a.stride(1),
               b.stride(0), b.stride(1),
               c.stride(0), c.stride(1),
               BM=128, BN=128, BK=32, runs=10)
    flops = 2 * M * N * K
    bytes = (M * K + K * N) * 2 + M * N * 4
    ai = flops / bytes
    gflops = flops / ms / 1e6
    results.append(('matmul_4096', ms, flops, bytes, ai, 0))
    print(f"  matmul 4096³: {ms:.3f} ms, {gflops:.0f} GFLOP/s, AI={ai:.1f}")

    # ── matmul @ 512 ──
    M = N = K = 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    ms = bench(matmul_kernel, g2,
               a, b, c, M, N, K,
               a.stride(0), a.stride(1),
               b.stride(0), b.stride(1),
               c.stride(0), c.stride(1),
               BM=128, BN=128, BK=32, runs=10)
    flops = 2 * M * N * K
    bytes = (M * K + K * N) * 2 + M * N * 4
    ai = flops / bytes
    gflops = flops / ms / 1e6
    results.append(('matmul_512', ms, flops, bytes, ai, 0))
    print(f"  matmul 512³: {ms:.3f} ms, {gflops:.0f} GFLOP/s, AI={ai:.1f}")

    # ── 打印汇总 ──
    print("\n" + "-" * 80)
    print(f"{'Kernel':<20} {'Time(ms)':<10} {'GFLOP/s':<12} {'GB/s':<10} {'AI':<10}")
    print("-" * 80)
    for name, ms, flops, bytes_, ai, gbps in results:
        g = flops / ms / 1e6 if flops > 0 else 0
        b = bytes_ / ms / 1e6 if bytes_ > 0 else gbps
        a = ai if ai > 0 else 0
        print(f"{name:<20} {ms:<10.3f} {g:<12.0f} {b:<10.0f} {a:<10.1f}")

    # ── 画 roofline ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # ceilings
    bw_peak = max(r[5] for r in results if r[5] > 0)  # use measured BW
    compute_peaks = {
        'FP32': 2861,  # from Step 01 C++
        'TC FP16': 12527,
        'TF32': 10932,
        'INT8': 46700,
    }
    colors = ['blue', 'red', 'green', 'orange']
    ai_range = np.logspace(-1, 5, 100)
    for (name, peak), color in zip(compute_peaks.items(), colors):
        mem = bw_peak * ai_range
        comp = np.full_like(ai_range, peak)
        roof = np.minimum(mem, comp)
        label = f'{name}: {peak/1000:.1f} TFLOP/s' if peak >= 1000 else f'{name}: {peak:.0f} GFLOP/s'
        ax.loglog(ai_range, roof, color=color, linestyle='--', alpha=0.7, label=label)
        ridge = peak / bw_peak
        ax.axvline(ridge, color=color, linestyle=':', alpha=0.2)

    # data points
    markers = {'copy': 's', 'saxpy': 'v', 'fma_x100': '^',
               'matmul_512': 'o', 'matmul_1024': 'o', 'matmul_4096': 'o'}
    colors_m = {'copy': 'green', 'saxpy': 'cyan', 'fma_x100': 'purple',
                'matmul_512': 'red', 'matmul_1024': 'orange', 'matmul_4096': 'magenta'}
    for name, ms, flops, bytes_, ai, gbps in results:
        g = flops / ms / 1e6 if flops > 0 else 0
        b = bytes_ / ms / 1e6 if bytes_ > 0 else gbps
        a = g / b if g > 0 and b > 0 else 0
        marker = markers.get(name, 'o')
        color = colors_m.get(name, 'blue')
        ax.scatter(max(a, 1e-3), g, marker=marker, c=color, s=100, edgecolors='black', zorder=5)
        label = f"{name} ({g:.0f} GFLOP/s)"
        ax.annotate(label, (max(a, 1e-3), g), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (GFLOP/s)')
    ax.set_title(f'Triton Roofline — {torch.cuda.get_device_name(device)} (BW={bw_peak:.0f} GB/s)')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('roofline_triton.png', dpi=150)
    print(f"\nRoofline 已保存: roofline_triton.png")


if __name__ == "__main__":
    main()
