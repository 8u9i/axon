# Axon Benchmark Suite

Direct comparison of Axon runtime vs SafeTensors for model weight loading.

## Setup

```bash
# Generate test models
pip install safetensors numpy
python generate_models.py

# Build and run Axon benchmarks
cargo build --release -p axon-runtime
./target/release/axon-bench --model results/model_100tensors_1gb.axon

# Run SafeTensors benchmarks
python bench_safetensors.py results/model_100tensors_1gb.safetensors
```

## Benchmark Models

Generated models cover representative scenarios:

| Model | Tensors | Size | DTypes |
|-------|---------|------|--------|
| Small  | 10     | 10MB | F16, F32 |
| Medium | 100    | 100MB | F16, F32 |
| Large  | 1000   | 1GB  | F16, F32 |
| XL     | 10000  | 10GB | F16, F32 |

## Metrics Measured

| Metric | Description |
|--------|-------------|
| Open Time | Time to mmap + parse headers |
| First Tensor Access | Time to access first tensor after open |
| Random Tensor Access | Time to access a random tensor |
| 100 Random Accesses | Time to access 100 random tensors |
| Full Tensor Scan | Time to sequentially scan all tensors |
| Partial Tensor Access | Time to access a partial byte range |
| Peak Memory Usage | Maximum RSS during benchmark |

## Running

```bash
# Full benchmark suite (requires safetensors installed)
./run_all.sh

# Axon only
cargo test --release -p axon-runtime --bench runtime -- --bench

# SafeTensors only
python bench_safetensors.py results/*.safetensors
```

## Success Criteria

Axon should demonstrate at least one measurable advantage:

- Lower peak memory usage than SafeTensors
- Faster first tensor access
- Better partial tensor access performance
- Lower startup latency
- Scalable behavior as model size grows

## Results Directory

Raw results are stored in `results/` with format:

```
results/
├── model_small_results.json
├── model_medium_results.json
├── ...
└── latest/
    └── summary.txt
```
