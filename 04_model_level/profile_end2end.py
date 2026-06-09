#!/usr/bin/env python3
"""End-to-end model profiling: run inference, parse ncu data, generate report."""

import subprocess
import json
import re
import sys
from pathlib import Path


def run_ncu_profile(binary: str, output_file: str = "profile_output.txt"):
    ncu = "/usr/local/cuda-13.0/bin/ncu"
    cmd = [
        ncu, "--target-processes", "all",
        "--launch-skip", "0",
        "--launch-count", "1",
        "--set", "basic",
        "--print-summary", "per-kernel",
        "-o", output_file.replace(".txt", ""),
        binary
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    Path(output_file).write_text(output)
    return output


def run_nsys_profile(binary: str):
    nsys = "/usr/local/cuda-13.0/bin/nsys"
    output_name = f"profile_{Path(binary).name}"
    cmd = [
        nsys, "profile",
        "--trace=cuda,nvtx",
        "--output", output_name,
        "--force-overwrite=true",
        binary
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    return f"{output_name}.nsys-rep"


def parse_kernel_times(output: str) -> list[dict]:
    kernels = []
    current = {}
    for line in output.split('\n'):
        if line.startswith("Kernel:"):
            if current:
                kernels.append(current)
            current = {"name": line.split("Kernel:")[1].strip()}
        elif "Duration" in line and current:
            m = re.search(r"Duration:\s+([\d.]+)\s+(m?s)", line)
            if m:
                current["duration"] = float(m.group(1))
                current["duration_unit"] = m.group(2)
        elif "Throughput" in line and current:
            m = re.search(r"(DRAM|SM|L1|L2)\s+Throughput", line)
            if m:
                val_m = re.search(r"([\d.]+)\s+([\w/]+)", line)
                if val_m:
                    current[f"{m.group(1).lower()}_throughput"] = float(val_m.group(1))
    if current:
        kernels.append(current)
    return kernels


def generate_report(kernels: list[dict], total_time_ms: float = None):
    print("\n" + "=" * 60)
    print("PROFILING REPORT")
    print("=" * 60)

    if total_time_ms:
        print(f"\nTotal inference time: {total_time_ms:.3f} ms")

    if not kernels:
        print("\nNo kernel data found. Run with ncu --print-summary per-kernel")
        return

    print(f"\n{'Kernel':<40} {'Duration(ms)':<15} {'% of Total':<15}")
    print("-" * 70)

    total = sum(k.get("duration", 0) for k in kernels)
    for k in sorted(kernels, key=lambda x: x.get("duration", 0), reverse=True):
        d = k.get("duration", 0)
        pct = (d / total * 100) if total > 0 else 0
        name = k["name"][:38]
        print(f"{name:<40} {d:<15.4f} {pct:<15.1f}")

    if total > 0:
        print("-" * 70)
        print(f"{'TOTAL':<40} {total:<15.4f} {100.0:<15.1f}")

    slowest = max(kernels, key=lambda x: x.get("duration", 0))
    print(f"\nSlowest kernel: {slowest['name']} ({slowest.get('duration', 0):.4f} ms)")
    if "dram_throughput" in slowest or "sm_throughput" in slowest:
        print("Key metrics:")
        for key in ["dram_throughput", "sm_throughput", "occupancy", "mem_pipes_busy"]:
            if key in slowest:
                print(f"  {key}: {slowest[key]}")


if __name__ == "__main__":
    binary = "./04_model_level/mini_cnn"

    if "--nsys" in sys.argv:
        rep = run_nsys_profile(binary)
        print(f"\nNsight Systems report: {rep}")
        print("View with: nsys-ui " + rep)
        sys.exit(0)

    print("Running ncu profiling (this may take a minute)...")
    output = run_ncu_profile(binary)

    kernels = parse_kernel_times(output)
    generate_report(kernels)

    print("\n" + "=" * 60)
    print("Quick analysis:")
    mem_kernels = [k for k in kernels if k.get("dram_throughput", 0) > 100]
    compute_kernels = [k for k in kernels if k.get("sm_throughput", 0) > 50]

    if mem_kernels:
        print(f"Memory-heavy kernels: {len(mem_kernels)}")
    if compute_kernels:
        print(f"Compute-heavy kernels: {len(compute_kernels)}")

    print("\nTo view full report: cat profile_output.txt")
