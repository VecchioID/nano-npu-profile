"""
PyTorch CUDA Graph 实战
======================
对比三种模式：
  1. 传统 eager mode
  2. torch.compile (reduce-overhead)
  3. 手写 CUDAGraph

运行: python3 demo_pytorch.py
"""

import torch
import torch.nn as nn
import time

# ── 一个小模型（模拟 mini_cnn 那种多小 kernel 的场景）──
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)  # 假设输入 32x32

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def benchmark(model, x, label, runs=200, warmup=20):
    """测推理吞吐。"""
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    inf_per_s = runs / elapsed
    ms_per_inf = elapsed / runs * 1000
    print(f"  {label:<30} {ms_per_inf:.3f} ms/inf, {inf_per_s:>8.0f} inf/s")
    return inf_per_s


def main():
    print("=" * 60)
    print("PyTorch CUDA Graph 实战")
    print("=" * 60)
    print()

    device = torch.cuda.current_device()
    print(f"设备: {torch.cuda.get_device_name(device)}")
    print()

    model = TinyModel().cuda().eval()
    x = torch.randn(1, 3, 32, 32, device='cuda')

    # ── 1. 传统 Eager Mode ──
    print("── 模式 1: 传统 Eager ──")
    inf1 = benchmark(model, x, "eager mode")

    # ── 2. torch.compile(reduce-overhead) ──
    print("\n── 模式 2: torch.compile (reduce-overhead) ──")
    compiled = torch.compile(model, mode="reduce-overhead")
    inf2 = benchmark(compiled, x, "compile(reduce-overhead)")
    print(f"  torch.compile 加速: {inf2/inf1:.2f}x")

    # ── 3. 手写 CUDA Graph ──
    print("\n── 模式 3: 手写 CUDA Graph ──")
    # 预热（让模型稳定）
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

    # 捕获 graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = model(x)
    torch.cuda.synchronize()

    # 重放
    runs = 200
    warmup = 20
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    inf_per_s = runs / elapsed
    ms_per_inf = elapsed / runs * 1000
    print(f"  CUDA Graph (手写)                {ms_per_inf:.3f} ms/inf, {inf_per_s:>8.0f} inf/s")
    print(f"  CUDA Graph 加速: {inf_per_s/inf1:.2f}x")

    # ── 4. 不同 batch size 的对比 ──
    print("\n── Batch Size 对 CUDA Graph 加速的影响 ──")
    for bs in [1, 2, 4, 8, 16]:
        x_bs = torch.randn(bs, 3, 32, 32, device='cuda')

        # eager
        for _ in range(10): model(x_bs)
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(100): model(x_bs)
        torch.cuda.synchronize()
        eager_ms = (time.time() - s) / 100 * 1000

        # graph
        graph_bs = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_bs):
            y_bs = model(x_bs)
        torch.cuda.synchronize()
        for _ in range(10): graph_bs.replay()
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(100): graph_bs.replay()
        torch.cuda.synchronize()
        graph_ms = (time.time() - s) / 100 * 1000

        speedup = eager_ms / graph_ms
        print(f"  batch={bs:<2}: eager {eager_ms:.3f} ms → graph {graph_ms:.3f} ms, "
              f"加速 {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("结论:")
    print("  - CUDA Graph 对小 batch 推理加速明显")
    print("  - batch 越大，加速效果越小（因为 kernel 执行时间占主导）")
    print("  - torch.compile 一行代码就能得到大部分收益")
    print("=" * 60)


if __name__ == "__main__":
    main()
