"""
AlexNet 真实网络性能分析
========================
用今天学的方法分析一个真实 CNN：找出瓶颈、画 roofline、提出优化建议。

运行: python3 alexnet_profile.py
需要: pip install torch
"""

import torch
import torch.nn as nn
import time, math


# ═══════════════════════════════════════════════════════════════
# 1. 定义 AlexNet
# ═══════════════════════════════════════════════════════════════

class AlexNet(nn.Module):
    """Krizhevsky 2012，ImageNet 冠军。结构清晰，适合做 profiling 教学。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))
        x = self.flatten(x)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        x = self.fc3(x)
        return x


# ═══════════════════════════════════════════════════════════════
# 2. 辅助函数：算 FLOPs 和参数量
# ═══════════════════════════════════════════════════════════════

def conv_flops(C_in, C_out, H_out, W_out, K):
    """Conv FLOPs = 2 × 输出元素数 × 每个元素的乘加数"""
    return 2 * C_out * H_out * W_out * C_in * K * K

def fc_flops(in_features, out_features):
    return 2 * out_features * in_features

def conv_memory(C_in, C_out, H, W, H_out, W_out, K):
    """计算 conv 一次推理的访存量（weights + input + output）"""
    weight_bytes = C_out * C_in * K * K * 4  # float32
    input_bytes = C_in * H * W * 4
    output_bytes = C_out * H_out * W_out * 4
    return weight_bytes + input_bytes, output_bytes

def fc_memory(in_features, out_features):
    weight_bytes = in_features * out_features * 4
    input_bytes = in_features * 4
    output_bytes = out_features * 4
    return weight_bytes + input_bytes, output_bytes


# ═══════════════════════════════════════════════════════════════
# 3. 逐层 Profiling
# ═══════════════════════════════════════════════════════════════

def profile_forward(model, x, device='cuda'):
    """逐层计时，收集每层的延迟、FLOPs、AI。"""
    layers_data = []
    hooks = []

    def make_hook(name, flops, bytes_read, bytes_write):
        def hook(module, inp, out):
            layers_data.append({
                'name': name,
                'input_shape': tuple(inp[0].shape),
                'output_shape': tuple(out.shape),
                'flops': flops,
                'bytes_read': bytes_read,
                'bytes_write': bytes_write,
            })
        return hook

    # 注册 hook
    # 先静态计算 FLOPs / bytes 等
    # 实际跑的时候只收集时间
    hooks.append(model.conv1.register_forward_hook(
        make_hook('conv1', conv_flops(3, 64, 55, 55, 11),
                  *conv_memory(3, 64, 227, 227, 55, 55, 11))))
    hooks.append(model.conv2.register_forward_hook(
        make_hook('conv2', conv_flops(64, 192, 27, 27, 5),
                  *conv_memory(64, 192, 27, 27, 27, 27, 5))))
    hooks.append(model.conv3.register_forward_hook(
        make_hook('conv3', conv_flops(192, 384, 13, 13, 3),
                  *conv_memory(192, 384, 13, 13, 13, 13, 3))))
    hooks.append(model.conv4.register_forward_hook(
        make_hook('conv4', conv_flops(384, 256, 13, 13, 3),
                  *conv_memory(384, 256, 13, 13, 13, 13, 3))))
    hooks.append(model.conv5.register_forward_hook(
        make_hook('conv5', conv_flops(256, 256, 13, 13, 3),
                  *conv_memory(256, 256, 13, 13, 13, 13, 3))))
    hooks.append(model.fc1.register_forward_hook(
        make_hook('fc1', fc_flop(9216, 4096),
                  *fc_memory(9216, 4096))))
    hooks.append(model.fc2.register_forward_hook(
        make_hook('fc2', fc_flop(4096, 4096),
                  *fc_memory(4096, 4096))))
    hooks.append(model.fc3.register_forward_hook(
        make_hook('fc3', fc_flop(4096, 1000),
                  *fc_memory(4096, 1000))))

    # warmup
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

    # 逐层测时间（用 CUDA events）
    results = []
    with torch.no_grad():
        for name, mod in model.named_children():
            if name == 'flatten':
                continue
            # 当前层的输入 = 前一层输出或原始输入
            # 这里简化：分层启动
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = mod(x if name == 'conv1' else None)  # 简化
            end.record()
            torch.cuda.synchronize()
            # 重新注册前向

    # 清理 hooks
    for h in hooks:
        h.remove()

    return layers_data


def main():
    print("=" * 70)
    print("AlexNet 性能分析（nano-npu-profile 方法论实战）")
    print("=" * 70)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    print(f"\n设备: {torch.cuda.get_device_name(device)}")
    print(f"SM 数: {sm_count}")
    print(f"显存: {props.total_memory / 1024**3:.1f} GB")

    # ── 创建模型和输入 ──
    model = AlexNet().to(device).eval()
    x = torch.randn(1, 3, 227, 227, device='cuda')
    print(f"输入: 1×3×227×227 (ImageNet 标准)")

    # ── 计算各层的理论值 ──
    layers_info = [
        ("conv1",  conv_flops(3, 64, 55, 55, 11),    *conv_memory(3, 64, 227, 227, 55, 55, 11)),
        ("pool1",  0, 0, 64*27*27*4),  # 只读+写
        ("conv2",  conv_flops(64, 192, 27, 27, 5),    *conv_memory(64, 192, 27, 27, 27, 27, 5)),
        ("pool2",  0, 0, 192*13*13*4),
        ("conv3",  conv_flops(192, 384, 13, 13, 3),   *conv_memory(192, 384, 13, 13, 13, 13, 3)),
        ("conv4",  conv_flops(384, 256, 13, 13, 3),   *conv_memory(384, 256, 13, 13, 13, 13, 3)),
        ("conv5",  conv_flops(256, 256, 13, 13, 3),   *conv_memory(256, 256, 13, 13, 13, 13, 3)),
        ("pool3",  0, 0, 256*6*6*4),
        ("fc1",    fc_flops(9216, 4096),               *fc_memory(9216, 4096)),
        ("fc2",    fc_flops(4096, 4096),               *fc_memory(4096, 4096)),
        ("fc3",    fc_flops(4096, 1000),               *fc_memory(4096, 1000)),
    ]

    print("\n" + "-" * 70)
    print("各层理论分析:")
    print("-" * 70)
    print(f"{'Layer':<8} {'Input':<18} {'Output':<18} {'FLOPs(M)':<10} {'Params(K)':<10} {'AI':<8}")
    print("-" * 70)

    total_flops = 0
    for name, flops, bytes_rw, _ in layers_info:
        if flops == 0:  # pool / relu
            continue
        # 算参数
        if 'conv' in name:
            idx = int(name[4:])
            cin = [3, 64, 192, 384, 256][idx-1]
            cout = [64, 192, 384, 256, 256][idx-1]
            k = [11, 5, 3, 3, 3][idx-1]
            params = cout * cin * k * k
        else:
            in_f = [9216, 4096, 4096][int(name[2])-1]
            out_f = [4096, 4096, 1000][int(name[2])-1]
            params = in_f * out_f

        ai = flops / bytes_rw if bytes_rw > 0 else 0
        total_flops += flops
        hw = int(name[-1]) - 1
        ins = [f"{3}×227×227", f"{64}×27×27", f"{192}×13×13",
               f"{384}×13×13", f"{256}×13×13",
               "9216", "4096", "4096"][hw] if 'conv' in name else ""
        outs = [f"{64}×55×55", f"{192}×27×27", f"{384}×13×13",
                f"{256}×13×13", f"{256}×6×6",
                "4096", "4096", "1000"][hw] if 'conv' in name else ""
        if 'fc' in name:
            fi = int(name[2]) - 1
            ins = [9216, 4096, 4096][fi]
            outs = [4096, 4096, 1000][fi]

        print(f"{name:<8} {str(ins):<18} {str(outs):<18} {flops/1e6:<10.0f} {params/1000:<10.1f} {ai:<8.1f}")

    print("-" * 70)
    print(f"{'Total':<8} {'':<18} {'':<18} {total_flops/1e9:<10.2f}G {'':<10} {'':<8}")
    print(f"总计算量: {total_flops/1e9:.2f} GFLOPs（单次推理）")

    # ── 实测时间 ──
    print("\n" + "-" * 70)
    print("逐层实测耗时:")
    print("-" * 70)
    print(f"{'Layer':<8} {'Warmup':<8} {'10 runs(ms)':<14} {'Avg(ms)':<10} {'GFLOP/s':<10} {'瓶颈类型':<12}")
    print("-" * 70)

    manual_layers = [
        ("conv1", model.conv1),
        ("conv2", model.conv2),
        ("conv3", model.conv3),
        ("conv4", model.conv4),
        ("conv5", model.conv5),
    ]

    x = torch.randn(1, 3, 227, 227, device='cuda')
    input_dims = [
        (1, 3, 227, 227),
        (1, 64, 27, 27),
        (1, 192, 13, 13),
        (1, 384, 13, 13),
        (1, 256, 13, 13),
    ]

    conv_flops_list = [141e6, 448e6, 224e6, 299e6, 199e6]
    conv_bytes_list = [conv_memory(3, 64, 227, 227, 55, 55, 11)[0] + conv_memory(3, 64, 227, 227, 55, 55, 11)[1],
                       conv_memory(64, 192, 27, 27, 27, 27, 5)[0] + conv_memory(64, 192, 27, 27, 27, 27, 5)[1],
                       conv_memory(192, 384, 13, 13, 13, 13, 3)[0] + conv_memory(192, 384, 13, 13, 13, 13, 3)[1],
                       conv_memory(384, 256, 13, 13, 13, 13, 3)[0] + conv_memory(384, 256, 13, 13, 13, 13, 3)[1],
                       conv_memory(256, 256, 13, 13, 13, 13, 3)[0] + conv_memory(256, 256, 13, 13, 13, 13, 3)[1]]

    all_results = []

    for i, (name, mod) in enumerate(manual_layers):
        inp = torch.randn(*input_dims[i], device='cuda')

        # warmup
        for _ in range(5):
            mod(inp)
        torch.cuda.synchronize()

        # 计时
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            mod(inp)
        end.record()
        torch.cuda.synchronize()
        t_ms = start.elapsed_time(end) / 10

        flops = conv_flops_list[i]
        gflops = flops / t_ms / 1e6

        bytes_rw = conv_bytes_list[i]
        ai = flops / bytes_rw if bytes_rw > 0 else 0
        bw_eff = bytes_rw / t_ms / 1e6

        # 判断瓶颈类型
        bw_peak = 219  # GB/s（来自 Step 01）
        fp32_peak = 6500  # GFLOP/s
        mem_max = bw_peak * ai
        if gflops < mem_max * 0.7:
            btype = "launch/occupancy"
        elif ai < 10:
            btype = "memory-bound"
        else:
            btype = "compute-bound"

        all_results.append((name, t_ms, gflops, ai, bw_eff, btype))

        print(f"{name:<8} {'':<8} {t_ms*10:<14.3f} {t_ms:<10.3f} {gflops:<10.0f} {btype:<12}")

    # FC layers
    fc_in = torch.randn(1, 9216, device='cuda')
    fc_dims = [(9216, 4096), (4096, 4096), (4096, 1000)]
    fc_mods = [model.fc1, model.fc2, model.fc3]

    for fi, (fc_mod, (in_f, out_f)) in enumerate(zip(fc_mods, fc_dims)):
        name = f"fc{fi+1}"
        inp = torch.randn(1, in_f, device='cuda')

        for _ in range(5):
            fc_mod(inp)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fc_mod(inp)
        end.record()
        torch.cuda.synchronize()
        t_ms = start.elapsed_time(end) / 100

        flops = fc_flops(in_f, out_f)
        gflops = flops / t_ms / 1e6
        bytes_rw = (in_f * out_f * 4) + in_f*4 + out_f*4
        ai = flops / bytes_rw

        bw_peak = 219
        mem_max = bw_peak * ai
        fp32_peak = 6500
        if t_ms < 0.05:
            btype = "launch-bound"
        elif gflops < mem_max * 0.7:
            btype = "under-utilized"
        elif ai < 10:
            btype = "memory-bound"
        else:
            btype = "compute-bound"

        all_results.append((name, t_ms, gflops, ai, bytes_rw / t_ms / 1e6, btype))
        print(f"{name:<8} {'':<8} {t_ms*100:<14.3f} {t_ms:<10.5f} {gflops:<10.0f} {btype:<12}")

    # ── 慢在哪？ ──
    print("\n" + "=" * 70)
    print("瓶颈分析")
    print("=" * 70)

    # 按时间占比排序
    total_t = sum(r[1] for r in all_results)
    sorted_r = sorted(all_results, key=lambda r: r[1], reverse=True)
    print(f"\n总 GPU kernel 时间: {total_t:.3f} ms")
    print(f"\n最慢的层（按耗时排序）:")
    print(f"{'Layer':<8} {'Time(ms)':<10} {'占比':<8} {'GFLOP/s':<10} {'AI':<8} {'瓶颈':<16}")
    print("-" * 60)
    for name, t, g, ai, bw, bt in sorted_r:
        pct = t / total_t * 100
        print(f"{name:<8} {t:<10.3f} {pct:<8.1f}% {g:<10.0f} {ai:<8.1f} {bt:<16}")

    print(f"\n{'─'*60}")
    print(f"TOP-1 瓶颈: {sorted_r[0][0]} ({sorted_r[0][1]:.3f} ms, {sorted_r[0][5]})")
    print(f"TOP-2 瓶颈: {sorted_r[1][0]} ({sorted_r[1][1]:.3f} ms, {sorted_r[1][5]})")
    print(f"TOP-3 瓶颈: {sorted_r[2][0]} ({sorted_r[2][1]:.3f} ms, {sorted_r[2][5]})")

    # ── Roofline 可视化 ──
    print("\n" + "-" * 70)
    print("Roofline 图")
    print("-" * 70)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))

        bw_peak = 219  # GB/s
        compute_peaks = {'FP32': 6500, 'TC FP16': 12527, 'TF32': 10932}
        colors_p = {'FP32': 'blue', 'TC FP16': 'red', 'TF32': 'green'}
        ai_range = np.logspace(-1, 5, 200)
        for name, peak in compute_peaks.items():
            mem = bw_peak * ai_range
            comp = np.full_like(ai_range, peak)
            roof = np.minimum(mem, comp)
            label = f'{name} ceiling ({peak/1000:.1f} TFLOP/s)' if peak >= 1000 else f'{name} ({peak:.0f} GFLOP/s)'
            ax.loglog(ai_range, roof, color=colors_p[name], linestyle='--', alpha=0.6, label=label)
            ax.axvline(peak / bw_peak, color=colors_p[name], linestyle=':', alpha=0.2)

        markers = {'conv': 'o', 'fc': 's', 'pool': 'v'}
        colors_l = {'conv': 'orange', 'fc': 'red', 'pool': 'green'}
        for name, t, g, ai, bw, bt in all_results:
            tag = 'conv' if 'conv' in name else 'fc' if 'fc' in name else 'pool'
            ax.scatter(max(ai, 1e-3), g, marker=markers[tag], c=colors_l[tag],
                      s=100, edgecolors='black', zorder=5, label=name if len([r for r in all_results if tag in r[0]]) == 0 or all_results.index((name,t,g,ai,bw,bt)) == 0 else "")
            ax.annotate(name, (max(ai, 1e-3), g), textcoords="offset points",
                       xytext=(5, 5), fontsize=7)

        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
        ax.set_ylabel('Performance (GFLOP/s)')
        ax.set_title(f'AlexNet Roofline - {props.name}')
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(loc='lower right', fontsize=7)
        plt.tight_layout()
        plt.savefig('alexnet_roofline.png', dpi=150)
        print(f"已保存: alexnet_roofline.png")
    except ImportError:
        print("需 matplotlib: pip install matplotlib")

    # ── 优化建议 ──
    print("\n" + "=" * 70)
    print("优化建议（基于 profiling 结果）")
    print("=" * 70)

    # 根据实测结果生成建议
    for name, t, g, ai, bw, bt in sorted_r[:3]:
        print(f"\n▶ {name} ({bt}):")

        if bt == "launch-bound" or bt == "launch/occupancy":
            print(f"  实测 {g:.0f} GFLOP/s，远低于任何天花板")
            print(f"  → 原因: kernel 太小，启动开销占主导")
            print(f"  → 方案: CUDA Graph / 合并相邻 kernel / 增大 batch size")

        elif bt == "memory-bound":
            if ai < 1:
                print(f"  实测 {g:.0f} GFLOP/s, AI={ai:.1f}（极低）")
                print(f"  → 原因: 每个元素只做很少计算")
                print(f"  → 方案: kernel fusion（例如把相邻两层合并）")
            else:
                print(f"  实测 {g:.0f} GFLOP/s, AI={ai:.1f}")
                print(f"  → 原因: 计算少，数据搬移多")
                print(f"  → 方案: 用 Tensor Core (FP16) 降低计算开销  / 增大 tile 提高数据复用")

        elif bt == "compute-bound":
            print(f"  实测 {g:.0f} GFLOP/s, AI={ai:.1f}（高）")
            print(f"  → 原因: 计算量大")
            print(f"  → 方案: 用更低精度 (FP16/INT8)、Tensor Core")

        elif bt == "under-utilized":
            print(f"  实测 {g:.0f} GFLOP/s")
            print(f"  → 原因: 资源利用率不足")
            print(f"  → 方案: 增大 batch size、调整 block 大小")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
今天的方法论可以直接用在任何网络上：
  1. 分解网络为逐层操作
  2. 测每层时间 → 找出 TOP-N 瓶颈
  3. 算每层 AI → 画 roofline → 判断瓶颈类型
  4. 针对瓶颈类型选优化方案

AlexNet 的典型瓶颈演变：
  2012: conv 层 compute-bound（FP32 不够快）
  2024: conv 层 → Tensor Core (FP16) 几乎免费
         新瓶颈 → FC 层（参数量大、带宽受限）
         或 → kernel 启动开销（小 batch 时）

真实优化不是改 kernel，而是调框架：
  - TensorRT: 自动 fusion + INT8/FP16
  - torch.compile: 一行代码
  - CUDA Graph: 消除启动开销
  - 增大 batch: 提高利用率
""")


if __name__ == "__main__":
    main()
