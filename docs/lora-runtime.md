# LoRA Runtime

## Purpose

Side-load LoRA adapters as separate `.axon` patches without modifying
the base model file. The runtime applies patch tensors over base tensors
at access time.

## Architecture

```
base.axon (stable, never modified)
    +
adapter.axon (LoRA weights)
    ↓
PatchedRuntime
    ↓
tensor_view("q_proj") → patch tensor if available, else base tensor
```

## Public API

```rust
use axon_runtime::{AxonRuntime, PatchedRuntime};

let base = AxonRuntime::open("model.axon")?;
let mut runtime = PatchedRuntime::new(base);

// Auto-detect LoRA naming: "lora_q_proj" overrides "q_proj"
runtime.attach("adapter.axon")?;

// Manual mapping
runtime.attach_raw("adapter.axon", vec![("lora_q_proj".into(), "q_proj".into())])?;

// Access tensor — patch if available, base otherwise
let data = runtime.tensor("q_proj")?;
```

## Patch Priority

```
Latest attached patch tensor
    ↓ (if not found)
Older patch tensor
    ↓ (if not found)
Base tensor
```

## LoRA Naming Convention

Adapters use `lora_` prefixes that are stripped to find the base tensor:

```
adapter.axon     →  base.axon
lora_q_proj      →  q_proj
lora_k_proj      →  k_proj
lora_v_proj      →  v_proj
lora_o_proj      →  o_proj
```

## Current Status

Implemented with 6 tests verifying:
- Auto-attach with LoRA naming convention
- Manual attach with explicit mappings
- Active patch selection
- Merge with context manager
- Detach all patches
- Base fallback when patch tensor is missing

## Limitations

- Only `PatchStrategy::Override` is implemented (patch replaces base).
  `PatchStrategy::Add` (element-wise addition) is planned.
- Patches must have the same shape as the base tensor.
- All patches are kept in memory (no lazy loading for patches yet).

## Future Work

- `PatchStrategy::Add` for element-wise LoRA application
- Lazy-loaded patches (only load patch tensors on access)
- Multiple active patches simultaneously
- Patch verification (signature, compatibility checks)
