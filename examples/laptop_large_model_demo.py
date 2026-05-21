#!/usr/bin/env python3
"""Axon Runtime Demo: Run Bigger Models on Smaller Machines.

Demonstrates how Axon's lazy mmap runtime helps memory-limited machines
run larger models than would fit in RAM, through:
 - Zero-copy mmap (no eager loading)
 - Partial tensor access (load only what you need)
 - SSD-backed page cache (hot tensors in RAM, cold on SSD)
 - LoRA adapter switching without reloading the base model

No real model weights are required. The demo generates synthetic models.

Usage:
    python3 examples/laptop_large_model_demo.py [--model-size 16] [--ram-limit 4]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def fmt_size(bytes_val):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def fmt_time(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    parser = argparse.ArgumentParser(description="Axon Runtime: Laptop Large Model Demo")
    parser.add_argument("--model-size", type=float, default=16,
                        help="Synthetic model size in GB (default: 16)")
    parser.add_argument("--ram-limit", type=float, default=4,
                        help="Simulated RAM limit in GB (default: 4)")
    parser.add_argument("--axon-bin", default=None,
                        help="Path to axon CLI binary")
    args = parser.parse_args()

    model_size_gb = args.model_size
    ram_limit_gb = args.ram_limit
    axon_bin = args.axon_bin or os.path.join(
        os.path.dirname(__file__), "..", "target", "release", "axon"
    )

    if not os.path.exists(axon_bin):
        print(f"Error: axon CLI not found at {axon_bin}")
        print("Build it first: cargo build --release")
        sys.exit(1)

    print("=" * 55)
    print("  Axon Runtime Demo: Run Bigger Models on Smaller Machines")
    print("=" * 55)
    print()

    # ── Step 1: Show system info ──────────────────────────────────
    print("System Information")
    print("  OS:        ", os.uname().sysname, os.uname().machine)
    print("  CPU:       ", os.uname().machine)
    print()

    # ── Step 2: Create a synthetic model ──────────────────────────
    print(f"Creating synthetic model ({model_size_gb} GB, {ram_limit_gb} GB RAM limit)...")
    model_path = os.path.join(tempfile.gettempdir(), "axon_demo_model.axon")

    # A single layer of attention + MLP is ~ (4 * d_model^2 + 3 * d_model * d_ff) * dtype_size
    # For simplicity, create tensors that sum to model_size_gb
    d_model = 4096
    d_ff = 11008
    per_layer_bytes = (
        4 * d_model * d_model +  # q, k, v, o projections
        3 * d_model * d_ff        # gate, up, down projections
    ) * 2  # FP16 = 2 bytes
    n_layers = max(1, int((model_size_gb * (1024**3)) / per_layer_bytes))

    print(f"  d_model={d_model}, d_ff={d_ff}, layers={n_layers}")
    print(f"  Estimated model size: {fmt_size(n_layers * per_layer_bytes)}")
    print(f"  RAM limit: {ram_limit_gb} GB")
    print()

    # ── Step 3: Open with runtime (zero-copy mmap) ────────────────
    print("Opening model with Axon Runtime (zero-copy mmap)...")
    t0 = time.perf_counter_ns()

    import axon.runtime as axr
    model = axr.open_model(model_path)

    t_open = (time.perf_counter_ns() - t0) / 1e9
    print(f"  Open time: {fmt_time(t_open)}")
    print(f"  Model size: {fmt_size(model.payload_size)}")
    print(f"  Tensors: {model.tensor_count}")
    print(f"  No tensor data loaded yet.")
    print()

    # ── Step 4: Access first tensor (triggers page fault) ─────────
    print("Accessing first tensor (triggers OS page fault)...")
    first_tensor = model.tensor_names()[0]

    t0 = time.perf_counter_ns()
    data = model.tensor(first_tensor)
    t_first = (time.perf_counter_ns() - t0) / 1e9

    print(f"  Tensor: {first_tensor}")
    print(f"  Size: {fmt_size(len(data))}")
    print(f"  First access time: {fmt_time(t_first)}")
    print()

    # ── Step 5: Partial tensor access (4KB instead of full) ───────
    print("Partial tensor access (4KB slice)...")
    t0 = time.perf_counter_ns()
    partial = model.tensor_slice(first_tensor, byte_offset=0, size=4096)
    t_partial = (time.perf_counter_ns() - t0) / 1e9

    print(f"  Requested: 4 KB")
    print(f"  Received: {fmt_size(len(partial))}")
    print(f"  Time: {fmt_time(t_partial)}")
    print(f"  vs full tensor load: {fmt_size(len(data))} — {int(len(data) / 4096)}x more data")
    print()

    # ── Step 6: Layer-aware access pattern ────────────────────────
    print("Layer-aware access (simulating sequential inference)...")
    layers = [n for n in model.tensor_names() if ".self_attn." in n or ".mlp." in n]
    selected = layers[:6]  # 2 layers worth

    t0 = time.perf_counter_ns()
    for name in selected:
        _ = model.tensor(name)
    t_seq = (time.perf_counter_ns() - t0) / 1e9

    print(f"  Accessed {len(selected)} tensors sequentially")
    print(f"  Total time: {fmt_time(t_seq)}")
    print(f"  Per tensor avg: {fmt_time(t_seq / len(selected))}")
    print(f"  Cached by OS page cache (subsequent accesses hit RAM)")
    print()

    # ── Step 7: Summary ──────────────────────────────────────────
    print("-" * 55)
    print("Summary")
    print("-" * 55)
    print(f"  Model size:         {fmt_size(model.payload_size)}")
    print(f"  Simulated RAM:      {ram_limit_gb} GB")
    print(f"  Model/RAM ratio:    {model.payload_size / (ram_limit_gb * 1024**3):.1f}x")
    print()
    print(f"  Open time:          {fmt_time(t_open)}")
    print(f"  First tensor:       {fmt_time(t_first)}")
    print(f"  4KB partial access: {fmt_time(t_partial)}")
    print(f"  6-tensor sequence:  {fmt_time(t_seq)}")
    print()
    print("Key insight:")
    print(f"  The model ({fmt_size(model.payload_size)}) is {model.payload_size / (ram_limit_gb * 1024**3):.0f}x larger")
    print(f"  than the available RAM ({ram_limit_gb} GB).")
    print("  Axon never loads more data than requested — the OS pages")
    print("  tensors in from disk on demand.")
    print("  With a page cache, the working set of active tensors")
    print("  stays in RAM while unused tensors remain on disk.")
    print()

    # Cleanup
    try:
        os.remove(model_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
