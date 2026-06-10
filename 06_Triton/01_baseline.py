"""
Step 01 Triton 版：测天花板
===========================
对应 C++ 版 Step 01：
  - copy_kernel  → 测带宽 (GB/s)
  - fma_kernel   → 测算力 (GFLOP/s)
  - tc_kernel    → Tensor Core 纯吞吐

运行：python3 01_baseline.py
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════
# 1. 带宽测试：float4 copy
# ═══════════════════════════════════════════════════════════════

@triton.jit
def copy_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def fma_kernel(data_ptr, n, iters: tl.constexpr, BLOCK: tl.constexpr):
    """每个元素反复做 FMA（乘加），测计算吞吐。"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(data_ptr + offsets, mask=mask)
    for _ in range(iters):
        x = x * 1.05 + 0.01
    tl.store(data_ptr + offsets, x, mask=mask)


@triton.jit
def tc_throughput_kernel(in_ptr, out_ptr, M, N, K, BLOCK: tl.constexpr):
    """从 global memory 读数据做 tl.dot 循环，测 Tensor Core 极限吞吐。
    读真实数据确保编译器不优化掉 tl.dot。"""
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)

    # 只读一次数据
    a = tl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :])
    b = tl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :])
    c = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    for _ in range(K // BLOCK):
        c += tl.dot(a, b)

    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], c)


def format_gbps(bytes_, ms):
    return bytes_ / ms / 1e6  # bytes / (ms*1e-3) / 1e9 = bytes/ms/1e6


def format_tflops(flops, ms):
    return flops / ms / 1e9  # flops / (ms*1e-3) / 1e12 = flops/ms/1e9


def main():
    print("=" * 60)
    print("Step 01 — Triton 版：测天花板")
    print("=" * 60)

    device = torch.cuda.current_device()
    print(f"\n设备: {torch.cuda.get_device_name(device)}")
    print(f"SM 数: {torch.cuda.get_device_properties(device).multi_processor_count}")

    total_elems = 64 * 1024 * 1024
    bytes_total = total_elems * 4 * 2  # read + write
    print(f"\n数据量: {total_elems:,} floats = {total_elems * 4 / 1024**3:.2f} GB")

    # ── 1. 带宽 ──
    print("\n--- 1. 带宽测试 (float copy) ---")
    x = torch.randn(total_elems, device='cuda', dtype=torch.float32)
    y = torch.empty(total_elems, device='cuda', dtype=torch.float32)

    grid_fn = lambda meta: (triton.cdiv(total_elems, meta['BLOCK']),)

    # warmup
    for _ in range(3):
        copy_kernel[grid_fn](x, y, total_elems, BLOCK=1024)
    torch.cuda.synchronize()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        copy_kernel[grid_fn](x, y, total_elems, BLOCK=1024)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 100
    bw = format_gbps(bytes_total, ms)
    print(f"  copy: {ms:.4f} ms avg, {bw:.0f} GB/s")

    # ── 2. 计算吞吐 (FMA) ──
    print("\n--- 2. 计算吞吐 (FP32 FMA, 10K iters) ---")
    z = torch.randn(total_elems, device='cuda', dtype=torch.float32)

    for _ in range(3):
        fma_kernel[grid_fn](z, total_elems, iters=10000, BLOCK=1024)
    torch.cuda.synchronize()

    start.record()
    for _ in range(10):
        fma_kernel[grid_fn](z, total_elems, iters=10000, BLOCK=1024)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    flops_per_elem = 2 * 10000  # mul + add per iter
    total_flops = total_elems * flops_per_elem
    ms_fma = ms
    print(f"  FMA x10000: {ms:.2f} ms avg, {format_tflops(total_flops, ms):.2f} TFLOPS")

    # ── 3. Tensor Core 吞吐 ──
    print("\n--- 3. Tensor Core 纯吞吐 (tl.dot) ---")
    TC_M = 16
    TC_N = 16
    TC_K = 1024 * 100  # 100 次 tl.dot per block
    tc_src = torch.randn(TC_M, TC_N, device='cuda', dtype=torch.float16)
    tc_out = torch.empty(TC_M, TC_N, device='cuda', dtype=torch.float32)
    tc_grid = (20,)  # 20 blocks = fill all SMs

    for _ in range(3):
        tc_throughput_kernel[tc_grid](tc_src, tc_out, TC_M, TC_N, TC_K, BLOCK=TC_M)
    torch.cuda.synchronize()

    start.record()
    for _ in range(10):
        tc_throughput_kernel[tc_grid](tc_src, tc_out, TC_M, TC_N, TC_K, BLOCK=TC_M)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    # Each block: (K/BLOCK) * (2*BLOCK^3) FLOPs = 100 * 8192 = 819,200 FLOPs
    flops_per_block = (TC_K // TC_M) * 2 * TC_M * TC_M * TC_M
    total_tc_flops = flops_per_block * tc_grid[0]
    print(f"  tl.dot x{(TC_K // TC_M)}: {ms:.3f} ms avg, {format_tflops(total_tc_flops, ms):.2f} TFLOPS")

    print("\n" + "=" * 60)
    print("结果对比 C++ 版:")
    print("  C++ TF32 copy:  219 GB/s")
    print(f"  Triton copy:    {bw:.0f} GB/s")
    print("  C++ FMA:        6.5 TFLOPS")
    print(f"  Triton FMA:     {format_tflops(total_flops, ms_fma):.2f} TFLOPS")
    print("  C++ TC FP16:    50.1 TFLOPS")
    print(f"  Triton tl.dot:  {format_tflops(total_tc_flops, ms):.2f} TFLOPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
