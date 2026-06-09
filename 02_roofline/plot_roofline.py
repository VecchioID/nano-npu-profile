#!/usr/bin/env python3
import subprocess, re, sys, math
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy"])
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

def parse_roofline_output(binary: str) -> tuple[list[dict], dict]:
    result = subprocess.run([binary], capture_output=True, text=True, timeout=120)
    kernels = []
    peak = {}
    pattern = r"([\w\s()+-]+)\s+\|\s+([\d.]+)\s+ms\s+\|\s+([\d.]+)\s+GFLOP/s\s+\|\s+([\d.]+)\s+GB/s\s+\|\s+([\d.]+)\s+FLOP/Byte"
    for line in result.stdout.split('\n'):
        m = re.search(r'BW_peak=([\d.]+).*?GFLOP_peak_fp32=([\d.]+).*?GFLOP_peak_tc16=([\d.]+).*?GOP_peak_int8=([\d.]+).*?GFLOP_peak_tf32=([\d.]+).*?WMMA_matmul_1024=([\d.]+)', line)
        if m:
            peak = {'bw': float(m.group(1)), 'fp32': float(m.group(2)),
                    'tc16': float(m.group(3)), 'int8': float(m.group(4)),
                    'tf32': float(m.group(5)), 'wmma': float(m.group(6))}
        m2 = re.search(pattern, line)
        if m2:
            kernels.append({
                'name': m2.group(1).strip(),
                'time_ms': float(m2.group(2)),
                'gflops': float(m2.group(3)),
                'gbps': float(m2.group(4)),
                'ai': float(m2.group(5)),
            })
    return kernels, peak

def plot_roofline(kernels, peak):
    fig, ax = plt.subplots(figsize=(12, 8))
    ai_range = np.logspace(-2, 6, 200)

    bw = peak['bw']

    ceilings = [
        ('FP32 CUDA Core', peak['fp32'], 'blue', '-'),
        ('TF32 Tensor Core', peak['tf32'], 'cyan', '-'),
        ('INT8 Tensor Core', peak['int8'], 'orange', '--'),
        ('FP16 Tensor Core', peak['tc16'], 'red', '-'),
    ]

    for name, gflops, color, style in ceilings:
        ridge = gflops / bw
        mem = bw * ai_range
        comp = np.full_like(ai_range, gflops)
        roof = np.minimum(mem, comp)
        label = f'{name}: {gflops/1000:.1f} TFLOP/s' if gflops >= 1000 else f'{name}: {gflops:.0f} GFLOP/s'
        ax.loglog(ai_range, roof, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=label)
        ax.axvline(ridge, color=color, linestyle=':', alpha=0.3)

    for k in kernels:
        if 'copy' in k['name']:
            color, marker, sz = 'green', 's', 80
            gf = peak['bw'] * 0.5
            label = f"{k['name']} ({k['gbps']:.0f} GB/s)"
        elif 'tc_fp16' in k['name']:
            color, marker, sz = 'red', 'D', 100
            gf = k['gflops']
            label = k['name']
        elif 'wmma_fp16' in k['name']:
            color, marker, sz = 'magenta', 'X', 120
            gf = k['gflops']
            label = k['name']
        else:
            color, marker, sz = 'blue', 'o', 80
            gf = k['gflops']
            label = k['name']
        ax.scatter(max(k['ai'], 1e-3), gf, marker=marker, c=color,
                   s=sz, edgecolors='black', zorder=5)
        ax.annotate(label, (max(k['ai'], 1e-3), gf),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    # FP8/FP4 estimates annotation
    ax.annotate('FP8 TC ~ 100 TFLOP/s (est.)\nFP4 TC ~ 200 TFLOP/s (est.)\n(tcgen05 not exposed on sm_110)',
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (GFLOP/s)')
    ax.set_title(f'Roofline Model - NVIDIA Thor (BW={bw:.0f} GB/s, {len(kernels)} kernels)')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('roofline.png', dpi=150)
    print(f"Roofline plot saved to roofline.png")

if __name__ == "__main__":
    binary = "./02_roofline/roofline_gen"
    kernels, peak = parse_roofline_output(binary)
    if not kernels or not peak:
        print("No data found. Run 'make run' first.")
        sys.exit(1)
    plot_roofline(kernels, peak)
