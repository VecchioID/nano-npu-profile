#!/usr/bin/env python3
"""Profiling result analysis tool: parse ncu/nsys output and generate reports."""

import subprocess
import json
import re
import sys
from pathlib import Path


def run_ncu_metrics(binary: str) -> dict:
    """Run ncu and extract key metrics."""
    ncu = "/usr/local/cuda-13.0/bin/ncu"
    cmd = [
        ncu, "--target-processes", "all",
        "--launch-skip", "0", "--launch-count", "1",
        "--set", "basic",
        "--print-summary", "per-kernel",
        binary
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    text = result.stdout + result.stderr
    metrics = {}

    patterns = [
        (r"Duration\s+([\d.]+)\s+(m?s)", "duration"),
        (r"DRAM Throughput\s+([\d.]+)\s+([\w/]+)", "dram_throughput"),
        (r"SM Throughput\s+([\d.]+)\s+([\w/]+)", "sm_throughput"),
        (r"Achieved Occupancy\s+([\d.]+)", "occupancy"),
        (r"Theoretical Occupancy\s+([\d.]+)", "theoretical_occupancy"),
        (r"Mem Pipes Busy\s+([\d.]+)%", "mem_pipes_busy"),
        (r"SM Busy\s+([\d.]+)%", "sm_busy"),
        (r"Registers\s+(\d+)", "registers"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                metrics[key] = m.group(1)

    metrics["raw_output"] = text[-2000:]
    return metrics


def classify_bottleneck(metrics: dict) -> str:
    """Classify kernel as compute-bound or memory-bound."""
    mem_busy = metrics.get("mem_pipes_busy", 0)
    sm_busy = metrics.get("sm_busy", 0)

    if mem_busy > 60 and sm_busy < 40:
        return "MEMORY_BOUND"
    elif sm_busy > 60 and mem_busy < 40:
        return "COMPUTE_BOUND"
    elif mem_busy > 40 and sm_busy > 40:
        return "BALANCED"
    else:
        return "LOW_UTILIZATION"


def generate_report(binary: str) -> str:
    """Generate a markdown profiling report."""
    metrics = run_ncu_metrics(binary)
    bottleneck = classify_bottleneck(metrics)

    lines = [
        f"# Profiling Report: {Path(binary).name}",
        "",
        f"## Bottleneck: {bottleneck}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        if k != "raw_output":
            lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## Optimization Suggestions")
    if bottleneck == "MEMORY_BOUND":
        lines.append("- Use shared memory tiling (increase data reuse)")
        lines.append("- Ensure coalesced global memory access")
        lines.append("- Consider using CUDA Graph to reduce launch overhead")
    elif bottleneck == "COMPUTE_BOUND":
        lines.append("- Use Tensor Cores (if available)")
        lines.append("- Increase instruction-level parallelism")
        lines.append("- Try reducing precision (FP16/INT8)")
    elif bottleneck == "LOW_UTILIZATION":
        lines.append("- Increase block size for better occupancy")
        lines.append("- Launch more blocks to utilize all SMs")
        lines.append("- Reduce thread divergence")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <binary>")
        sys.exit(1)

    report = generate_report(sys.argv[1])
    print(report)

    out_path = f"report_{Path(sys.argv[1]).name}.md"
    Path(out_path).write_text(report)
    print(f"\nReport saved to {out_path}")
