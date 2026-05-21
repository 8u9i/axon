//! # Axon Runtime
//!
//! SSD-backed lazy mmap runtime for .axon model weights.
//!
//! Unlike `axon-core` (which loads the entire file into a `Vec<u8>`), the
//! runtime memory-maps the file and only touches the bytes you actually access.
//! The OS handles the lazy loading — tensor pages are faulted in from disk
//! on first access, not eagerly.
//!
//! ## Architecture
//!
//! - `AxonRuntime` — the main entry point. Opens an `.axon` file, parses its
//!   metadata (header, manifest, tensor descriptors), and provides tensor access
//!   backed by an mmap.
//! - `MmapStore` — owns the memory-mapped file and provides safe byte-range
//!   access. No tensor bytes are loaded into application memory until requested.
//! - `TensorInfo` — metadata about a tensor (name, dtype, shape, location in file).
//! - `RuntimeStats` — instrumentation counters.
//!
//! ## Example
//!
//! ```no_run
//! use axon_runtime::AxonRuntime;
//!
//! let rt = AxonRuntime::open("model.axon").unwrap();
//! println!("Model: {}", rt.model_name());
//! println!("Tensors: {}", rt.tensor_count());
//!
//! let data = rt.tensor("layer_0_q").unwrap();
//! println!("Tensor size: {} bytes", data.len());
//!
//! let info = rt.tensor_info("layer_0_q").unwrap();
//! println!("DType: {}, Shape: {:?}", info.dtype.name(), info.shape);
//! ```

mod mmap_store;
mod runtime;
mod slice;
pub mod tensor_cache;
mod stats;
pub mod lora;
pub mod paging;

pub use mmap_store::MmapStore;
pub use runtime::{AxonRuntime, CachedRuntime, TensorInfo, TensorAccess};
pub use slice::{SliceSpec, TensorSlice};
pub use stats::RuntimeStats;
