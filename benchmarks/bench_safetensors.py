#!/usr/bin/env python3
"""SafeTensors benchmark for comparison with Axon."""

import sys
import time
import json
import os
import random
import resource
import safetensors
import safetensors.safe_open
import numpy as np

def get_memory_mb():
    """Get current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def bench_safetensors(path):
    """Benchmark SafeTensors loading from a file."""
    file_size = os.path.getsize(path)
    print(f"\n=== SafeTensors Benchmark ===")
    print(f"File: {path}")
    print(f"File size: {file_size / (1024**2):.2f} MB")
    mem_before = get_memory_mb()

    # ── Open ──
    t0 = time.perf_counter()
    with safetensors.safe_open(path, framework="np", device="cpu") as f:
        open_time = time.perf_counter() - t0
        print(f"\n[Open] {open_time * 1000:.2f} ms")

        names = list(f.keys())
        print(f"  Tensors: {len(names)}")

        # Get metadata for all tensors
        total_bytes = 0
        for name in names:
            total_bytes += f.get_tensor(name).nbytes

        # ── First tensor access ──
        first = names[0]
        t0 = time.perf_counter()
        tensor = f.get_tensor(first)
        first_access = time.perf_counter() - t0
        print(f"\n[First tensor access] {first} ({first_access * 1000:.2f} ms)")

        # ── Random tensor access ──
        mid = names[len(names) // 2]
        t0 = time.perf_counter()
        tensor = f.get_tensor(mid)
        random_access = time.perf_counter() - t0
        print(f"\n[Random tensor access] {mid} ({random_access * 1000:.2f} ms)")

        # ── 100 random accesses ──
        indices = list(range(min(100, len(names))))
        t0 = time.perf_counter()
        for i in indices:
            _ = f.get_tensor(names[i])
        hundred_access = time.perf_counter() - t0
        print(f"\n[100 tensor accesses] {hundred_access * 1000:.2f} ms ({hundred_access / 100 * 1000:.3f} ms avg)")

        # ── Full tensor scan ──
        t0 = time.perf_counter()
        for name in names:
            _ = f.get_tensor(name)
        full_scan = time.perf_counter() - t0
        throughput = total_bytes / full_scan / (1024**3) if full_scan > 0 else 0
        print(f"\n[Full tensor scan] {full_scan * 1000:.2f} ms")
        print(f"  Total accessed: {total_bytes / (1024**2):.2f} MB")
        print(f"  Throughput: {throughput:.2f} GB/s")

        # ── Partial tensor access (numpy slicing) ──
        if len(names) > 1:
            name = names[len(names) // 2]
            full_tensor = f.get_tensor(name)
            slice_size = min(full_tensor.nbytes // 4, 4096)
            t0 = time.perf_counter()
            partial = full_tensor.ravel()[:slice_size].copy()
            partial_time = time.perf_counter() - t0
            print(f"\n[Partial tensor access] {partial_time * 1000:.2f} ms ({slice_size} bytes)")
            print("  Note: SafeTensors loads the full tensor then slices — no true partial load")

        # ── Memory ──
        mem_after = get_memory_mb()
        print(f"\n[Peak memory] {mem_after:.3f} MB (delta: {mem_after - mem_before:.1f} MB)")

    # ── Comparison summary ──
    results = {
        "path": path,
        "file_size_bytes": file_size,
        "tensor_count": len(names),
        "total_data_bytes": total_bytes,
        "open_time_ms": open_time * 1000,
        "first_access_ms": first_access * 1000,
        "random_access_ms": random_access * 1000,
        "hundred_access_ms": hundred_access * 1000,
        "full_scan_ms": full_scan * 1000,
        "peak_memory_mb": mem_after,
    }

    print(f"\n{json.dumps(results, indent=2)}")
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python bench_safetensors.py <model.safetensors> [...]")
        sys.exit(1)

    all_results = []
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        results = bench_safetensors(path)
        all_results.append(results)

    # Save combined results
    os.makedirs("results", exist_ok=True)
    with open("results/safetensors_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to results/safetensors_results.json")
    print(f"Compare with Axon: benchmark Axon results are in results/*.axon")


if __name__ == "__main__":
    main()
