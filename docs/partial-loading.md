# Partial Tensor Loading

## Purpose

Loading only the parts of a tensor that are needed, instead of the whole
thing. This is critical for large models where loading every weight matrix
into RAM upfront is impractical.

## Architecture

Axon provides two levels of partial loading:

1. **Byte-level**: `tensor_byte_view(name, range)` — raw byte ranges
2. **Shape-aware**: `tensor_rows(name, start, end)` — row-based slicing
   using dtype size, shape, and stride math

## Public API (Rust)

```rust
// Zero-copy byte range (no shape awareness)
pub fn tensor_byte_view(&self, name: &str, range: Range<usize>) -> AxonResult<&[u8]>;

// Shape-aware row slicing for 2D tensors
pub fn tensor_rows(&self, name: &str, start_row: usize, end_row: usize) -> AxonResult<&[u8]>;
```

## Row Computation

For a tensor with shape `[N, M]` and dtype size `S`:

```
row_stride = M * S
byte_offset = start_row * row_stride
slice_size = (end_row - start_row) * row_stride
```

The runtime validates:
- Tensor name exists
- Shape is at least 2D
- Row range is within bounds
- No integer overflow in stride computation

## Public API (Python)

```python
import axon.runtime as axr

model = axr.open("model.axon")

# Byte range
chunk = model.tensor_slice("weight", byte_offset=0, size=4096)

# Row range
rows = model.tensor_rows("weight", 0, 128)
```

Python copies data into `bytes` objects for compatibility. The Rust API
provides true zero-copy borrowed views.

## Current Status

Implemented. Both byte-level and shape-aware partial loading are tested
and working.

## Example Use Cases

- Loading LoRA adapter weights without loading the full base model
- Inspecting the first few rows of an embedding table
- Reading tensor metadata without touching weight data
- Layer-by-layer inference where only one layer's weights are needed
  at a time

## Limitations

- Only 2D row slicing is implemented. Multi-dimensional slicing is
  future work.
- No scatter/gather for non-contiguous rows. Multiple calls needed.

## Future Work

- N-dimensional slicing (not just rows)
- Column-based slicing
- Strided access (every Nth row)
- Integration with the `TensorPager` trait for SSD-backed partial
  loads of models larger than RAM
