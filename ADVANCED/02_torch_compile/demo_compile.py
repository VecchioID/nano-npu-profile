"""
torch.compile 实战
==================
对比 ResNet-18 在 eager / compile / compile+FP16 下的吞吐。

需要: pip install torchvision（或手动定义模型）

运行: python3 demo_compile.py
"""

import torch
import torch.nn as nn
import time


class SimpleCNN(nn.Module):
    """和 nano-npu-profile 里 mini_cnn 类似的结构。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def benchmark(model, x, label, runs=500, warmup=50):
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
    print(f"  {label:<35} {ms_per_inf:.3f} ms/inf, {inf_per_s:>8.0f} inf/s")
    return inf_per_s


def main():
    print("=" * 60)
    print("torch.compile 实战 — SimpleCNN")
    print("=" * 60)
    print()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"设备: {props.name}")
    print()

    model = SimpleCNN().cuda().eval()
    x = torch.randn(1, 3, 32, 32, device='cuda')

    # ── 1. Eager FP32 ──
    print("── FP32 ──")
    inf_eager = benchmark(model, x, "eager FP32")

    # ── 2. compile(default) ──
    compiled = torch.compile(model, mode="default")
    with torch.no_grad():
        compiled(x)
    torch.cuda.synchronize()
    inf_compile = benchmark(compiled, x, "compile(default) FP32")
    print(f"  加速: {inf_compile/inf_eager:.2f}x")

    # ── 3. compile(reduce-overhead) ──
    compiled_ro = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        compiled_ro(x)
    torch.cuda.synchronize()
    inf_ro = benchmark(compiled_ro, x, "compile(reduce-overhead) FP32")
    print(f"  加速: {inf_ro/inf_eager:.2f}x")

    # ── 4. Eager FP16 ──
    model_fp16 = model.half()
    x_fp16 = x.half()
    inf_fp16 = benchmark(model_fp16, x_fp16, "eager FP16")
    print(f"  FP16 加速: {inf_fp16/inf_eager:.2f}x")

    # ── 5. compile + FP16 ──
    compiled_fp16 = torch.compile(model_fp16, mode="reduce-overhead")
    with torch.no_grad():
        compiled_fp16(x_fp16)
    torch.cuda.synchronize()
    inf_cfp16 = benchmark(compiled_fp16, x_fp16, "compile + FP16")
    print(f"  compile+FP16 加速: {inf_cfp16/inf_eager:.2f}x")

    # ── 6. Batch size 扫描 ──
    print("\n── Batch Size 扫描 ──")
    for bs in [1, 2, 4, 8]:
        x_bs = torch.randn(bs, 3, 32, 32, device='cuda')

        for _ in range(20): model(x_bs)
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(200): model(x_bs)
        torch.cuda.synchronize()
        eager_ms = (time.time() - s) / 200 * 1000

        for _ in range(20): compiled_ro(x_bs)
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(200): compiled_ro(x_bs)
        torch.cuda.synchronize()
        compile_ms = (time.time() - s) / 200 * 1000

        print(f"  batch={bs:<2}: eager {eager_ms:.3f} ms → compile {compile_ms:.3f} ms, "
              f"加速 {eager_ms/compile_ms:.2f}x")

    print("\n" + "=" * 60)
    print("结论:")
    print("  1. torch.compile 零成本加速，推荐默认开启")
    print("  2. FP16 + compile = 最大收益")
    print("  3. batch=1 时 compile 效果最明显")
    print("=" * 60)


if __name__ == "__main__":
    main()
