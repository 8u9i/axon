pub mod checksum;
pub mod convert;
pub mod error;
pub mod header;
pub mod manifest;
pub mod mmap_loader;
pub mod tensor;

pub use checksum::*;
pub use error::*;
pub use header::*;
pub use manifest::*;
pub use mmap_loader::*;
pub use tensor::*;

pub const AXON_MAGIC: &[u8; 4] = b"AXON";
pub const AXON_VERSION: u32 = 1;
pub const CACHE_LINE_SIZE: u64 = 64;
