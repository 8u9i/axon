# Benchmarks

## Purpose

Reproducible, documented comparison of Axon runtime vs SafeTensors for
model weight loading performance.

## Running Benchmarks

```bash
# Generate test models
cd benchmarks
pip install safetensors numpy
python generate_models.py

# Run Axon benchmarks
cargo run --release --example bench_axon -- results/model_100tensors_100mb.axon

# Run SafeTensors benchmarks
python bench_safetensors.py results/model_100tensors_100mb.safetensors
```

## What We Measure

| Metric | Why It Matters |
|--------|---------------|
| File open time | Startup latency for inference servers |
| First tensor access | Time to access first weight after open |
| Random tensor access | Worst-case cold access latency |
| 100 random accesses | Realistic random access pattern |
| Full tensor scan | Sequential throughput bottleneck |
| Partial tensor access | LoRA/adapter loading scenario |
| Peak memory usage | RAM budget for edge devices |

## Success Criteria for Axon

At least one measurable advantage over SafeTensors:

1. **Lower peak memory** — mmap avoids eager full-file read
2. **Faster first tensor access** — lazy mmap only pages in what's needed
3. **Better partial tensor access** — shape-aware slicing loads less data
4. **Lower startup latency** — parsing only metadata, not data
5. **Scalable behavior** as model size grows beyond RAM

## Notes

- Axon benchmarks measure mmap-backed access — the OS manages page
  residency
- SafeTensors benchmarks measure full tensor reads into Python/numpy
  objects
- Fair comparison requires the same hardware and models
- Memory usage comparison is approximate (OS page cache effects vary)

## Current Status

Benchmark infrastructure is set up. Generated models cover 10MB to 1GB+.
Reproducible numbers require running on the target hardware with the
`benchmarks/` scripts.
