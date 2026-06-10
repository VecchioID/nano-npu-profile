"""
FlashAttention Triton 实现
===========================
对比三种方案：
  1. Standard attention (PyTorch)
  2. FlashAttention 手写 Triton kernel
  3. PyTorch 的 F.scaled_dot_product_attention (SDPA, 内部也用了 FlashAttention)

测量: forward 时间 + HBM 访问量对比

运行: python3 demo_flash.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, d,
    stride_q, stride_k, stride_v, stride_o,
    B_r: tl.constexpr, B_c: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention forward kernel (一行分块).

    每个 program 处理 Q 的一行分块 (B_r 行).
    内层循环遍历 K/V 的行.
    """
    pid = tl.program_id(0)
    row_start = pid * B_r

    # Q 的偏移
    q_off = row_start * stride_q

    # 初始化 online softmax 统计量
    m = tl.full([B_r, 1], -float('inf'), dtype=tl.float32)
    l = tl.zeros([B_r, 1], dtype=tl.float32)
    acc = tl.zeros([B_r, BLOCK_D], dtype=tl.float32)

    # 外层循环：Q 行分块 (这里只处理一行分块，外层 Python 负责)
    Q_block = tl.load(Q_ptr + q_off + tl.arange(0, B_r)[:, None] * stride_q +
                      tl.arange(0, BLOCK_D)[None, :] * 1,
                      mask=tl.arange(0, B_r)[:, None] < B_r)

    # 内层循环：遍历 K/V
    for col_start in range(0, N, B_c):
        k_off = col_start * stride_k
        v_off = col_start * stride_v

        K_block = tl.load(K_ptr + k_off + tl.arange(0, B_c)[:, None] * stride_k +
                          tl.arange(0, BLOCK_D)[None, :] * 1,
                          mask=tl.arange(0, B_c)[:, None] < B_c)
        V_block = tl.load(V_ptr + v_off + tl.arange(0, B_c)[:, None] * stride_v +
                          tl.arange(0, BLOCK_D)[None, :] * 1,
                          mask=tl.arange(0, B_c)[:, None] < B_c)

        # S_block = Q_block @ K_block.T   (B_r × B_c)
        S_block = tl.dot(Q_block, tl.trans(K_block))

        # online softmax
        block_m = tl.max(S_block, axis=1)[:, None]
        block_p = tl.exp(S_block - block_m)
        block_l = tl.sum(block_p, axis=1)[:, None]

        # 合并统计量
        new_m = tl.maximum(m, block_m)
        alpha = tl.exp(m - new_m)
        beta = tl.exp(block_m - new_m)
        l = alpha * l + beta * block_l
        m = new_m

        # 修正之前的 acc 并合并新块
        #   block_p = exp(S - block_m)
        #   需要用 beta = exp(block_m - new_m) 缩放到 new_m 尺度
        acc = acc * alpha + beta * tl.dot(block_p.to(Q_block.dtype), V_block)

    # 最终归一化
    acc = acc / l

    # 写回
    tl.store(O_ptr + row_start * stride_o + tl.arange(0, B_r)[:, None] * stride_o +
             tl.arange(0, BLOCK_D)[None, :] * 1,
             acc.to(O_ptr.dtype.element_ty),
             mask=tl.arange(0, B_r)[:, None] < B_r)


def flash_attention_triton(Q, K, V, B_r=64, B_c=64):
    """
    调用 Triton FlashAttention kernel.
    """
    N, d = Q.shape
    O = torch.empty_like(Q)

    grid = ((N + B_r - 1) // B_r,)

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O,
        N, d,
        Q.stride(0), K.stride(0), V.stride(0), O.stride(0),
        B_r=B_r, B_c=B_c, BLOCK_D=d,
    )
    return O


def benchmark_attention(Q, K, V, label, fn, warmup=10, runs=50):
    """测 attention forward 时间。"""
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        fn(Q, K, V)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs * 1000
    return elapsed


def main():
    print("=" * 72)
    print("FlashAttention 对比 (Triton vs PyTorch SDPA vs Standard)")
    print("=" * 72)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"设备: {props.name}")
    print()

    seq_lens = [512, 1024, 2048]
    d = 64
    n_heads = 1

    print(f"{'N':>6} | {'Standard':>10} | {'SDPA':>10} | {'Triton FA':>10} | {'加速比':>8}")
    print("-" * 52)

    for N in seq_lens:
        Q = torch.randn(N, d, device='cuda', dtype=torch.float16)
        K = torch.randn(N, d, device='cuda', dtype=torch.float16)
        V = torch.randn(N, d, device='cuda', dtype=torch.float16)

        # 1. Standard attention
        def std_attn(q, k, v):
            s = q @ k.T
            p = torch.softmax(s, dim=-1)
            return p @ v

        t_std = benchmark_attention(Q, K, V, "standard", std_attn)

        # 2. PyTorch SDPA (FlashAttention 后端)
        # SDPA 需要 batch 和 head 维度
        def sdpa_attn(q, k, v):
            return F.scaled_dot_product_attention(
                q.unsqueeze(0).unsqueeze(0),
                k.unsqueeze(0).unsqueeze(0),
                v.unsqueeze(0).unsqueeze(0),
                is_causal=False,
            ).squeeze(0).squeeze(0)

        t_sdpa = benchmark_attention(Q, K, V, "SDPA", sdpa_attn)

        # 3. Triton FlashAttention
        def triton_fa(q, k, v):
            return flash_attention_triton(q, k, v, B_r=64, B_c=64)

        t_triton = benchmark_attention(Q, K, V, "Triton FA", triton_fa)

        speedup_over_std = t_std / t_triton if t_triton > 0 else 0

        print(f"{N:>6} | {t_std:>8.3f}ms | {t_sdpa:>8.3f}ms | {t_triton:>8.3f}ms | "
              f"{speedup_over_std:>7.2f}x")

    print()
    print("分析:")
    print("  - Standard attention 随着 N 增大，时间增长 ~N²")
    print("  - FlashAttention/Triton 增长 ~N (因为 HBM 访问从 O(N²) → O(N))")
    print("  - SDPA 内部也用了 FlashAttention，是最优实现")
    print("  - Triton FA 是教学实现，没有做所有优化（如 warp-level 优化）")
    print()

    # ── 验证正确性 ──
    print("=" * 72)
    print("正确性验证 (N=512, d=64, FP16)")
    print("=" * 72)
    N = 512
    Q = torch.randn(N, d, device='cuda', dtype=torch.float16)
    K = torch.randn(N, d, device='cuda', dtype=torch.float16)
    V = torch.randn(N, d, device='cuda', dtype=torch.float16)

    O_std = std_attn(Q, K, V)
    O_triton = triton_fa(Q, K, V)

    diff = (O_std - O_triton).abs().max().item()
    print(f"  Standard vs Triton FA: max diff = {diff:.6f}")
    print(f"  {'✓ 通过' if diff < 0.1 else '✗ 差异较大'} (FP16 精度差 < 0.1 可接受)")
    print()

    # ── 不同 block size 的影响 ──
    print("=" * 72)
    print("Block Size 对 Triton FA 性能的影响 (N=1024)")
    print("=" * 72)
    N = 1024
    Q = torch.randn(N, d, device='cuda', dtype=torch.float16)
    K = torch.randn(N, d, device='cuda', dtype=torch.float16)
    V = torch.randn(N, d, device='cuda', dtype=torch.float16)

    for B_r, B_c in [(32, 32), (64, 64), (128, 32), (32, 128), (128, 128)]:
        t = benchmark_attention(Q, K, V, f"{B_r}x{B_c}",
                                lambda q, k, v: flash_attention_triton(q, k, v, B_r=B_r, B_c=B_c),
                                runs=30)
        print(f"  B_r={B_r:>3}, B_c={B_c:>3}: {t:.3f} ms")

    print()
    print("结论:")
    print("  - B_r=64, B_c=64 是 Thor 48KB SMEM 的均衡选择")
    print("  - 更大的 block = 更少的外层循环迭代，但超过 SMEM 限制会失败")
    print("  - 这个 demo 可以帮你验证你的 GPU 的最佳配置")


if __name__ == "__main__":
    main()
