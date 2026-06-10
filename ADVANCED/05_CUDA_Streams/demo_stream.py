"""
CUDA Streams Demo (PyTorch)
============================
用 PyTorch 演示 stream 对推理吞吐的影响。

对比:
  1. 单 stream (顺序处理多输入)
  2. 多 stream (并行处理多输入)
  3. Stream + async data loading

运行: python3 demo_stream.py
"""

import torch
import torch.nn as nn
import time


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def benchmark_single_stream(model, x, label, runs=200, warmup=20):
    """单 stream 基准。"""
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  {label:<40} {elapsed/runs*1000:.3f} ms/inf, {runs/elapsed:>8.0f} inf/s")


def main():
    print("=" * 65)
    print("CUDA Streams Demo (PyTorch)")
    print("=" * 65)
    print()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"设备: {props.name}")
    print()

    model = TinyModel().cuda().eval()

    # ── 1. Single stream baseline ──
    print("── 模式 1: 单个 stream (顺序推理) ──")
    x = torch.randn(1, 3, 32, 32, device='cuda')
    benchmark_single_stream(model, x, "单 stream, 1 输入")

    # ── 2. 多 stream (并行推理多个输入) ──
    print("\n── 模式 2: 多 stream (并行推理) ──")
    for num_streams in [2, 4]:
        # 每个 stream 处理一个 batch=1 的输入
        inputs = [torch.randn(1, 3, 32, 32, device='cuda')
                  for _ in range(num_streams)]
        outputs = [None] * num_streams

        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        # 预热
        for _ in range(10):
            for i in range(num_streams):
                with torch.cuda.stream(streams[i]):
                    outputs[i] = model(inputs[i])
        torch.cuda.synchronize()

        # 计时
        warmup, runs = 20, 100
        for _ in range(warmup):
            for i in range(num_streams):
                with torch.cuda.stream(streams[i]):
                    outputs[i] = model(inputs[i])
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(runs):
            for i in range(num_streams):
                with torch.cuda.stream(streams[i]):
                    outputs[i] = model(inputs[i])
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # 总处理了多少个推理
        total_infs = runs * num_streams
        print(f"  {num_streams} streams, {runs} 轮 "
              f"→ {total_infs} 次推理, {elapsed:.3f}s total, "
              f"{total_infs/elapsed:>8.0f} inf/s "
              f"({elapsed/runs*1000:.3f} ms/轮)")

    # ── 3. Overlap data loading + inference ──
    print("\n── 模式 3: Overlap Data Loading + Inference ──")
    # 模拟: stream1 加载数据 (memcpy), stream2 推理
    stream_load = torch.cuda.Stream()
    stream_infer = torch.cuda.Stream()

    # 准备 CPU 数据 (pinned memory)
    x_cpu = torch.randn(1, 3, 32, 32).pin_memory()
    x_gpu = torch.empty(1, 3, 32, 32, device='cuda')

    event = torch.cuda.Event()

    # 预热
    for _ in range(10):
        with torch.cuda.stream(stream_load):
            x_gpu.copy_(x_cpu, non_blocking=True)
            event.record(stream_load)
        with torch.cuda.stream(stream_infer):
            stream_infer.wait_event(event)
            y = model(x_gpu)
    torch.cuda.synchronize()

    # 计时
    warmup, runs = 20, 200
    for _ in range(warmup):
        with torch.cuda.stream(stream_load):
            x_gpu.copy_(x_cpu, non_blocking=True)
            event.record(stream_load)
        with torch.cuda.stream(stream_infer):
            stream_infer.wait_event(event)
            y = model(x_gpu)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        with torch.cuda.stream(stream_load):
            x_gpu.copy_(x_cpu, non_blocking=True)
            event.record(stream_load)
        with torch.cuda.stream(stream_infer):
            stream_infer.wait_event(event)
            y = model(x_gpu)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  overlap data load + infer: {elapsed/runs*1000:.3f} ms/inf, "
          f"{runs/elapsed:>8.0f} inf/s")

    print("\n" + "=" * 65)
    print("结论:")
    print("  1. 多 stream 并行推理多个独立输入可以提升吞吐")
    print("  2. 同步事件 (cuda.Event) 是 stream 间协同的关键")
    print("  3. 实际加速取决于硬件是否有空闲资源")
    print("  4. 查看效果: nsys timeline 上可以看到重叠的 kernel")
    print("=" * 65)


if __name__ == "__main__":
    main()
