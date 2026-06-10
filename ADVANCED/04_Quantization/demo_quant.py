"""
量化推理实战 (纯 Python 模拟，不依赖量化后端)
==============================================
因为 PyTorch 2.12 的量化后端在 ARM64 (Thor) 上不可用，
我们用纯 Python 演示量化核心流程。

对比:
  1. FP32 baseline
  2. INT8 模拟量化 (手动 scale + zero_point)
  3. 不同量化粒度 (per-tensor vs per-channel)

注意: 实际 INT8 Tensor Core 加速已在 Step 03 验证 (46.7 TOPS)

运行: python3 demo_quant.py
"""

import torch
import torch.nn as nn
import time


def quantize_tensor(t, num_bits=8, per_channel=False, dim=0):
    """
    手动量化张量。

    参数:
      t: FP32 张量
      num_bits: 量化位数 (默认 8)
      per_channel: 是否按通道量化
      dim: 通道维度

    返回:
      q: INT8 量化张量
      scale: 缩放因子
      zero_point: 零点
    """
    if per_channel:
        # per-channel: 每个输出通道单独 scale
        shape = [1] * t.dim()
        shape[dim] = t.shape[dim]
        t_flat = t.transpose(0, dim).reshape(t.shape[dim], -1)
        min_vals = t_flat.min(dim=1).values.reshape(shape)
        max_vals = t_flat.max(dim=1).values.reshape(shape)
    else:
        min_vals = t.min()
        max_vals = t.max()

    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    scale = (max_vals - min_vals) / (qmax - qmin)
    scale = scale.clamp(min=1e-10)  # 避免除零
    zero_point = qmin - min_vals / scale
    zero_point = zero_point.round().clamp(qmin, qmax).to(torch.int32).to(t.dtype)

    # 量化
    q = (t / scale + zero_point).round().clamp(qmin, qmax).to(torch.int8)
    return q, scale, zero_point


def dequantize_tensor(q, scale, zero_point):
    """反量化。"""
    return (q.float() - zero_point) * scale


def quantized_linear(x, w, b, w_scale, w_zp, x_scale, x_zp):
    """
    模拟 INT8 Linear 层。

    实际 INT8 GEMM 在 Tensor Core 上执行 (Step 03 的 IMMA)，
    这里用 FP32 模拟精度效果。
    """
    w_fp32 = dequantize_tensor(w, w_scale, w_zp)
    x_fp32 = dequantize_tensor(x, x_scale, x_zp)
    y = x_fp32 @ w_fp32.T
    if b is not None:
        y = y + b
    return y


def quantized_conv2d(x, w, b, w_scale, w_zp, x_scale, x_zp, stride=1, padding=1):
    """模拟 INT8 Conv2d 层。"""
    w_fp32 = dequantize_tensor(w, w_scale, w_zp)
    x_fp32 = dequantize_tensor(x, x_scale, x_zp)
    y = torch.nn.functional.conv2d(x_fp32, w_fp32, bias=b, stride=stride, padding=padding)
    return y


class TinyCNNFP32(nn.Module):
    """FP32 参考实现。"""
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


class TinyCNNINT8:
    """
    INT8 模拟推理。
    用 FP32 算但是只使用量化后的权重 (模拟精度损失)。
    """
    def __init__(self, fp32_model, per_channel=False):
        self.fp32_model = fp32_model
        self.per_channel = per_channel
        self._quantize_weights()
        self.buffers = {}  # 存激活值的 scale/zp (校准)

    def _quantize_weights(self):
        """量化权重。"""
        self.w_quant = {}
        for name, param in self.fp32_model.named_parameters():
            if 'weight' in name:
                q, s, zp = quantize_tensor(param.data, per_channel=self.per_channel, dim=0)
                self.w_quant[name] = (q, s, zp)

    def calibrate(self, x):
        """校准: 记录激活值范围。"""
        h = x
        self.buffers['conv1'] = h
        h = self.fp32_model.conv1(h)
        self.buffers['conv1_out'] = h
        h = self.fp32_model.relu1(h)
        h = self.fp32_model.pool1(h)
        self.buffers['pool1'] = h
        h = self.fp32_model.conv2(h)
        self.buffers['conv2_out'] = h
        h = self.fp32_model.relu2(h)
        h = self.fp32_model.pool2(h)
        h = h.view(h.size(0), -1)
        self.buffers['fc_input'] = h
        h = self.fp32_model.fc(h)
        self.buffers['fc_out'] = h

    def _make_scale(self, tensor_name):
        """从校准数据生成 scale/zp。"""
        t = self.buffers[tensor_name]
        q, s, zp = quantize_tensor(t, per_channel=False)
        return s, zp

    def forward(self, x):
        """模拟 INT8 推理。"""
        x_s, x_zp = self._make_scale('conv1')

        # conv1
        w_q, w_s, w_zp = self.w_quant['conv1.weight']
        x = quantized_conv2d(x, w_q, self.fp32_model.conv1.bias,
                             w_s, w_zp, x_s, x_zp)
        x_s, x_zp = self._make_scale('conv1_out')

        # relu + pool (不需要量化)
        x = self.fp32_model.relu1(x)
        x = self.fp32_model.pool1(x)
        x_s, x_zp = self._make_scale('pool1')

        # conv2
        w_q, w_s, w_zp = self.w_quant['conv2.weight']
        x = quantized_conv2d(x, w_q, self.fp32_model.conv2.bias,
                             w_s, w_zp, x_s, x_zp)
        x_s, x_zp = self._make_scale('conv2_out')

        x = self.fp32_model.relu2(x)
        x = self.fp32_model.pool2(x)
        x = x.view(x.size(0), -1)
        x_s, x_zp = self._make_scale('fc_input')

        # fc
        w_q, w_s, w_zp = self.w_quant['fc.weight']
        x = quantized_linear(x, w_q, self.fp32_model.fc.bias,
                             w_s, w_zp, x_s, x_zp)

        return x


def main():
    print("=" * 60)
    print("量化推理实战 (纯 Python 模拟)")
    print("=" * 60)
    print()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"设备: {props.name}")
    print()

    model_fp32 = TinyCNNFP32().cuda().eval()
    x = torch.randn(1, 3, 32, 32, device='cuda')

    # ── FP32 Baseline ──
    print("── FP32 Baseline (CUDA) ──")
    param_size = sum(p.numel() * p.element_size() for p in model_fp32.parameters())
    print(f"  参数大小: {param_size / 1024:.1f} KB")

    warmup, runs = 50, 200
    for _ in range(warmup): model_fp32(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs): model_fp32(x)
    torch.cuda.synchronize()
    t_fp32 = (time.time() - start) / runs * 1000
    print(f"  推理时间: {t_fp32:.3f} ms")
    with torch.no_grad():
        y_fp32 = model_fp32(x)

    # ── INT8 模拟 ──
    print("\n── INT8 模拟推理 ──")

    for granularity, label in [(False, "per-tensor"), (True, "per-channel")]:
        model_int8 = TinyCNNINT8(model_fp32, per_channel=granularity)

        # 校准 (因为权重已经量化，校准只需要确定激活值的 scale)
        model_int8.calibrate(x)

        y_int8 = model_int8.forward(x)
        diff = (y_fp32 - y_int8).abs().max().item()

        # 模拟的 INT8 参数大小 (权重 1 字节 + scale/zp 各 4 字节)
        n_params = sum(p.numel() for p in model_fp32.parameters())
        scale_overhead = 0
        if granularity:
            scale_overhead = 4 * 2 * 3  # 3 层, per-channel, scale + zp
        else:
            scale_overhead = 4 * 2 * 3  # 3 层, per-tensor
        int8_size = n_params * 1 + scale_overhead

        print(f"  INT8 {label:<12} 模拟参数大小: {int8_size / 1024:.1f} KB "
              f"(原 FP32: {param_size / 1024:.1f} KB, 压缩 {param_size / int8_size:.1f}x)")
        print(f"  精度损失 (max diff vs FP32): {diff:.4f}")

    # ── 不同 bit 宽度的精度对比 ──
    print("\n── Bit Width 对精度的影响 ──")
    for bits in [8, 6, 4]:
        q, s, zp = quantize_tensor(model_fp32.fc.weight.data, num_bits=bits, per_channel=True)
        w_fp32 = dequantize_tensor(q, s, zp)
        diff = (model_fp32.fc.weight.data - w_fp32).abs().max().item()
        mse = (model_fp32.fc.weight.data - w_fp32).pow(2).mean().item()
        print(f"  INT{bits:<2}: max diff = {diff:.4f}, MSE = {mse:.6f}")

    print("\n" + "=" * 60)
    print("结论:")
    print("  1. INT8 模型缩小 ~4x")
    print("  2. per-channel 量化精度更高 (权重每行分布不同)")
    print("  3. CUDA INT8 Tensor Core 加速已在 Step 03 实测: 46.7 TOPS")
    print("     (vs FP32 FMA: 6.5 TFLOPS)")
    print("  4. PyTorch 2.12+ 的量化已迁移到 torchao (https://github.com/pytorch/ao)")
    print("=" * 60)


if __name__ == "__main__":
    main()
