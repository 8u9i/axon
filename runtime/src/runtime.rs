//! # AxonRuntime
//!
//! The main entry point for runtime-loaded `.axon` files.
//!
//! ## Design
//!
//! `AxonRuntime` memory-maps an `.axon` file and parses only the structural
//! metadata (header, manifest, tensor descriptors). No tensor data is loaded
//! into application memory until `tensor()` or `tensor_bytes()` is called.
//!
//! The OS handles lazy loading: the first access to a tensor's byte range
//! triggers a page fault, loading the corresponding file pages from disk
//! into the page cache. Subsequent accesses hit RAM.
//!
//! ## Ownership
//!
//! The runtime owns the `MmapStore` (and thus the mmap handle). Tensor data
//! is returned as owned `Vec<u8>` so it outlives the runtime. Future versions
//! may add scoped zero-copy access for read-only inference pipelines.
//!
//! ## Thread safety
//!
//! Phase 1 is single-threaded (`&mut self` for tensor access that may cache).
//! Metadata access (`tensor_info`, `tensors`, `model_name`) is read-only and
//! can be shared.

use std::path::Path;
use std::sync::Arc;

use axon_core::header::AxonHeader;
use axon_core::manifest::Manifest;
use axon_core::tensor::TensorDescriptor;
use axon_core::{AxonError, AxonResult, CACHE_LINE_SIZE};

use crate::mmap_store::MmapStore;
use crate::stats::RuntimeStats;
use crate::tensor_cache::{CacheStats as TensorCacheStats, TensorCache};

/// Metadata about a tensor in the model - no data loaded.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Data type code (see `axon_core::DType`).
    pub dtype: axon_core::DType,
    /// Shape of the tensor.
    pub shape: Vec<u64>,
    /// Byte offset in the file where this tensor's data begins.
    pub data_offset: u64,
    /// Size of the tensor data in bytes.
    pub data_size: u64,
}

impl From<&TensorDescriptor> for TensorInfo {
    fn from(d: &TensorDescriptor) -> Self {
        Self {
            name: d.name_str().to_string(),
            dtype: d.dtype().unwrap_or(axon_core::DType::F32),
            shape: d.shape_vec(),
            data_offset: d.data_offset,
            data_size: d.data_size,
        }
    }
}

/// Trait for tensor access strategies.
///
/// The default implementation reads from the mmap and returns owned bytes.
/// Future implementations will check a cache first, or support paging.
pub trait TensorAccess {
    /// Get the raw bytes of a tensor by name.
    fn tensor_bytes(&self, name: &str) -> AxonResult<Vec<u8>>;

    /// Get a contiguous byte range from a tensor without loading the whole thing.
    /// `byte_offset` is relative to the start of the tensor data.
    fn tensor_byte_range(&self, name: &str, byte_offset: u64, size: u64) -> AxonResult<Vec<u8>>;
}

/// The main runtime for lazy-loaded `.axon` files.
///
/// ## Opening a file
///
/// ```no_run
/// use axon_runtime::AxonRuntime;
///
/// let rt = AxonRuntime::open("model.axon").unwrap();
/// ```
///
/// No tensor data is loaded during `open()`. Only the header (64 bytes),
/// manifest (variable, typically a few KB), and tensor descriptor table
/// (192 bytes per tensor) are parsed from the mmap.
#[derive(Debug)]
pub struct AxonRuntime {
    store: MmapStore,
    header: AxonHeader,
    manifest: Manifest,
    descriptors: Vec<TensorDescriptor>,
    stats: RuntimeStats,
}

impl AxonRuntime {
    /// Open an `.axon` file and parse its metadata.
    ///
    /// This memory-maps the file and parses:
    /// - Header (64 bytes)
    /// - Manifest (JSON, variable size)
    /// - Tensor descriptor table (192 bytes per tensor)
    ///
    /// No tensor data is loaded. Tensor access is lazy — bytes are faulted
    /// in from disk on first access.
    pub fn open<P: AsRef<Path>>(path: P) -> AxonResult<Self> {
        let store = MmapStore::open(&path)?;

        // Parse header from the mmap (zero-copy)
        let header_bytes = store
            .raw_slice(0, AxonHeader::HEADER_SIZE as u64)
            .ok_or_else(|| AxonError::UnexpectedEof {
                needed: AxonHeader::HEADER_SIZE as u64,
                available: store.len(),
            })?;
        let header = AxonHeader::from_bytes(header_bytes)?;

        // Parse manifest from the mmap (zero-copy JSON parsing)
        let manifest_bytes = store
            .raw_slice(header.manifest_offset, header.manifest_size)
            .ok_or_else(|| AxonError::UnexpectedEof {
                needed: header.manifest_offset + header.manifest_size,
                available: store.len(),
            })?;
        let manifest: Manifest = serde_json::from_slice(manifest_bytes)
            .map_err(|e| AxonError::InvalidManifest(e.to_string()))?;

        // Parse tensor descriptor table from the mmap
        let tdt_start = align_up(
            header.manifest_offset + header.manifest_size,
            CACHE_LINE_SIZE,
        );
        let tdt_size = header.tensor_count * TensorDescriptor::SIZE as u64;
        let descriptors = if tdt_size > 0 {
            let tdt_bytes =
                store
                    .raw_slice(tdt_start, tdt_size)
                    .ok_or_else(|| AxonError::UnexpectedEof {
                        needed: tdt_start + tdt_size,
                        available: store.len(),
                    })?;
            parse_descriptor_table(tdt_bytes, header.tensor_count as usize)?
        } else {
            Vec::new()
        };

        log::info!(
            "Opened .axon: {} tensors, {} total payload bytes",
            header.tensor_count,
            header.payload_size,
        );

        Ok(Self {
            store,
            header,
            manifest,
            descriptors,
            stats: RuntimeStats::new(),
        })
    }

    /// Get the model name from the manifest.
    pub fn model_name(&self) -> &str {
        self.manifest.model.as_deref().unwrap_or("")
    }

    /// Get the model architecture from the manifest.
    pub fn architecture(&self) -> &str {
        self.manifest.architecture.as_deref().unwrap_or("")
    }

    /// Get the number of tensors in the model.
    pub fn tensor_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Get the total payload size (sum of all tensor data sizes) in bytes.
    pub fn payload_size(&self) -> u64 {
        self.header.payload_size
    }

    /// Get the size of the mmap'd file in bytes.
    pub fn file_size(&self) -> u64 {
        self.store.len()
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.descriptors.iter().map(|d| d.name_str()).collect()
    }

    /// Get metadata about a tensor without loading its data.
    pub fn tensor_info(&self, name: &str) -> AxonResult<TensorInfo> {
        let desc = self.find_descriptor(name)?;
        Ok(TensorInfo::from(desc))
    }

    /// Get metadata about all tensors.
    pub fn tensors(&self) -> Vec<TensorInfo> {
        self.descriptors.iter().map(TensorInfo::from).collect()
    }

    /// Get the raw bytes of a tensor by name.
    ///
    /// This is the primary tensor access method. It reads the tensor's byte
    /// range from the mmap. On first access, this triggers a page fault —
    /// the OS loads the tensor data from disk into the page cache.
    ///
    /// Returns an owned `Vec<u8>` that outlives the runtime.
    pub fn tensor(&self, name: &str) -> AxonResult<Vec<u8>> {
        let desc = self.find_descriptor(name)?;
        let bytes = self.store.read_bytes(desc.data_offset, desc.data_size)?;
        self.stats.record_access(desc.data_size);
        Ok(bytes)
    }

    /// Get a contiguous byte range from a tensor without loading the whole thing.
    ///
    /// `byte_offset` is relative to the start of the tensor's data. This is
    /// the foundation for partial tensor loading and slicing.
    ///
    /// Example: to load the first 4KB of a weight matrix:
    ///
    /// ```no_run
    /// # use axon_runtime::AxonRuntime;
    /// # let rt = AxonRuntime::open("model.axon").unwrap();
    /// let first_4k = rt.tensor_byte_range("layer_0_q", 0, 4096).unwrap();
    /// ```
    pub fn tensor_byte_range(
        &self,
        name: &str,
        byte_offset: u64,
        size: u64,
    ) -> AxonResult<Vec<u8>> {
        let desc = self.find_descriptor(name)?;
        if byte_offset + size > desc.data_size {
            return Err(AxonError::UnexpectedEof {
                needed: desc.data_offset + byte_offset + size,
                available: desc.data_offset + desc.data_size,
            });
        }
        let bytes = self
            .store
            .read_bytes(desc.data_offset + byte_offset, size)?;
        self.stats.record_access(size);
        Ok(bytes)
    }

    /// Get a zero-copy view into a tensor's data.
    ///
    /// Returns a `&[u8]` slice that borrows directly from the mmap.
    /// No allocation, no copying — the OS pages in the tensor data
    /// from disk on first access.
    ///
    /// The returned slice is valid as long as the runtime is alive.
    ///
    /// ```no_run
    /// # use axon_runtime::AxonRuntime;
    /// # let rt = AxonRuntime::open("model.axon").unwrap();
    /// let view: &[u8] = rt.tensor_view("layer_0_q").unwrap();
    /// ```
    pub fn tensor_view(&self, name: &str) -> AxonResult<&[u8]> {
        let desc = self.find_descriptor(name)?;
        let slice = self
            .store
            .raw_slice(desc.data_offset, desc.data_size)
            .ok_or_else(|| AxonError::UnexpectedEof {
                needed: desc.data_offset + desc.data_size,
                available: self.store.len(),
            })?;
        self.stats.record_access(desc.data_size);
        Ok(slice)
    }

    /// Get a zero-copy view into a contiguous byte range of a tensor.
    ///
    /// No allocation, no copying. Only the requested bytes are mapped.
    ///
    /// ```no_run
    /// # use axon_runtime::AxonRuntime;
    /// # let rt = AxonRuntime::open("model.axon").unwrap();
    /// let view: &[u8] = rt.tensor_byte_view("layer_0_q", 0..4096).unwrap();
    /// ```
    pub fn tensor_byte_view(&self, name: &str, range: std::ops::Range<usize>) -> AxonResult<&[u8]> {
        let desc = self.find_descriptor(name)?;
        let byte_offset = range.start as u64;
        let size = (range.end - range.start) as u64;
        if byte_offset + size > desc.data_size {
            return Err(AxonError::UnexpectedEof {
                needed: desc.data_offset + byte_offset + size,
                available: desc.data_offset + desc.data_size,
            });
        }
        let slice = self
            .store
            .raw_slice(desc.data_offset + byte_offset, size)
            .ok_or_else(|| AxonError::UnexpectedEof {
                needed: desc.data_offset + byte_offset + size,
                available: self.store.len(),
            })?;
        self.stats.record_access(size);
        Ok(slice)
    }

    /// Get a zero-copy view of contiguous rows from a 2D tensor.
    ///
    /// Uses the tensor's dtype and shape to compute exact byte offsets.
    /// No allocation, no copying — only the requested rows are mapped.
    ///
    /// ```no_run
    /// # use axon_runtime::AxonRuntime;
    /// # let rt = AxonRuntime::open("model.axon").unwrap();
    /// let rows: &[u8] = rt.tensor_rows("emb_weight", 0, 128).unwrap();
    /// ```
    pub fn tensor_rows(&self, name: &str, start_row: usize, end_row: usize) -> AxonResult<&[u8]> {
        let desc = self.find_descriptor(name)?;
        let dtype = desc.dtype()?;
        let shape = desc.shape_vec();

        if shape.len() < 2 {
            return Err(AxonError::InvalidManifest(format!(
                "tensor_rows requires at least 2D tensor, got {}D for '{}'",
                shape.len(),
                name
            )));
        }

        let cols = shape[1] as usize;
        let elem_size = dtype.size_in_bytes();
        let row_stride = cols
            .checked_mul(elem_size)
            .ok_or(AxonError::AlignmentError {
                offset: 0,
                alignment: 1,
            })?;
        let total_rows = shape[0] as usize;

        if start_row >= total_rows || end_row > total_rows || start_row > end_row {
            return Err(AxonError::UnexpectedEof {
                needed: desc.data_offset + (end_row * row_stride) as u64,
                available: desc.data_offset + desc.data_size,
            });
        }

        let byte_offset = start_row
            .checked_mul(row_stride)
            .ok_or(AxonError::AlignmentError {
                offset: 0,
                alignment: 1,
            })?;
        let num_rows = end_row - start_row;
        let size = num_rows
            .checked_mul(row_stride)
            .ok_or(AxonError::AlignmentError {
                offset: 0,
                alignment: 1,
            })?;

        let slice = self
            .store
            .raw_slice(desc.data_offset + byte_offset as u64, size as u64)
            .ok_or_else(|| AxonError::UnexpectedEof {
                needed: desc.data_offset + byte_offset as u64 + size as u64,
                available: self.store.len(),
            })?;

        self.stats.record_access(size as u64);
        Ok(slice)
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// Find a tensor descriptor by name.
    fn find_descriptor(&self, name: &str) -> AxonResult<&TensorDescriptor> {
        self.descriptors
            .iter()
            .find(|d| d.name_str() == name)
            .ok_or_else(|| AxonError::TensorNotFound(name.to_string()))
    }

    /// Access the underlying mmap store (for advanced use).
    pub fn store(&self) -> &MmapStore {
        &self.store
    }

    /// Access the parsed header.
    pub fn header(&self) -> &AxonHeader {
        &self.header
    }

    /// Access the parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}

impl TensorAccess for AxonRuntime {
    fn tensor_bytes(&self, name: &str) -> AxonResult<Vec<u8>> {
        self.tensor(name)
    }

    fn tensor_byte_range(&self, name: &str, byte_offset: u64, size: u64) -> AxonResult<Vec<u8>> {
        self.tensor_byte_range(name, byte_offset, size)
    }
}

// ── CachedRuntime ──────────────────────────────────────────────────

/// An `AxonRuntime` with an integrated LRU tensor cache.
///
/// When `tensor()` is called:
/// 1. Check the cache first (cache hit → return `Arc` clone, no copy)
/// 2. If not cached, read from the mmap, store in cache, return `Arc`
/// 3. If cache is full, evict LRU unpinned tensors until there's room
///
/// ## Example
///
/// ```no_run
/// use axon_runtime::AxonRuntime;
///
/// let mut rt = AxonRuntime::with_cache("model.axon", 4 * 1024 * 1024 * 1024).unwrap();
/// let data = rt.tensor_cached("layer_0_q").unwrap();
/// ```
#[derive(Debug)]
pub struct CachedRuntime {
    pub(crate) inner: AxonRuntime,
    pub(crate) cache: TensorCache,
}

impl CachedRuntime {
    /// Create a new cached runtime from an existing `AxonRuntime`.
    pub fn new(inner: AxonRuntime, cache_budget: usize) -> Self {
        Self {
            inner,
            cache: TensorCache::new(cache_budget),
        }
    }

    /// Get a tensor, using the cache.
    ///
    /// Cache hit: returns `Arc` clone (no copy).
    /// Cache miss: reads from mmap, stores in cache, returns `Arc`.
    pub fn tensor_cached(&mut self, name: &str) -> AxonResult<Arc<Vec<u8>>> {
        if let Some(cached) = self.cache.get(name) {
            self.inner.stats.record_cache_hit();
            return Ok(cached);
        }

        self.inner.stats.record_cache_miss();
        let bytes = self.inner.tensor(name)?;
        let size = bytes.len();
        let arc = self.cache.put(name.to_string(), bytes);
        log::debug!(
            "Cached tensor '{}' ({} bytes, usage: {}/{})",
            name,
            size,
            self.cache.current_usage(),
            self.cache.budget()
        );
        Ok(arc)
    }

    /// Pin a tensor in the cache (prevent eviction).
    pub fn pin(&mut self, name: &str) {
        self.cache.pin(name);
    }

    /// Unpin a tensor.
    pub fn unpin(&mut self, name: &str) {
        self.cache.unpin(name);
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> &TensorCacheStats {
        self.cache.stats()
    }

    /// Get the inner runtime (read-only).
    pub fn runtime(&self) -> &AxonRuntime {
        &self.inner
    }

    /// Get the inner runtime (mutable).
    pub fn runtime_mut(&mut self) -> &mut AxonRuntime {
        &mut self.inner
    }

    /// Evict a specific tensor from the cache.
    pub fn evict(&mut self, name: &str) {
        self.cache.evict(name);
    }

    /// Clear the cache entirely.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl AxonRuntime {
    /// Open a file with a cache of the given size (in bytes).
    ///
    /// This is a convenience constructor that creates a `CachedRuntime`.
    pub fn with_cache<P: AsRef<Path>>(path: P, cache_budget: usize) -> AxonResult<CachedRuntime> {
        let rt = Self::open(path)?;
        Ok(CachedRuntime::new(rt, cache_budget))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Align a value up to the given alignment.
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Parse a block of bytes as a table of `TensorDescriptor`s.
fn parse_descriptor_table(bytes: &[u8], count: usize) -> AxonResult<Vec<TensorDescriptor>> {
    let mut descriptors = Vec::with_capacity(count);
    let mut cursor = 0usize;
    for _ in 0..count {
        if cursor + TensorDescriptor::SIZE > bytes.len() {
            return Err(AxonError::UnexpectedEof {
                needed: (cursor + TensorDescriptor::SIZE) as u64,
                available: bytes.len() as u64,
            });
        }
        let desc = TensorDescriptor::from_bytes(&bytes[cursor..cursor + TensorDescriptor::SIZE])?;
        descriptors.push(desc);
        cursor += TensorDescriptor::SIZE;
    }
    Ok(descriptors)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use axon_core::{AxonBuilder, DType};

    /// Build a synthetic .axon file for testing.
    fn build_test_axon(path: &Path) {
        let mut builder = AxonBuilder::new().model("test-model").architecture("test");

        // 10 tensors with known values
        for i in 0..10 {
            let name = format!("layer_{}_weight", i);
            let data: Vec<u8> = (0..64).map(|j| (i * 64 + j) as u8).collect();
            builder = builder.add_tensor(&name, data, DType::U8, &[64]);
        }

        let bytes = builder.build().expect("Failed to build .axon");
        fs::write(path, &bytes).expect("Failed to write test file");
    }

    fn test_dir() -> PathBuf {
        let dir = PathBuf::from("output");
        fs::create_dir_all(&dir).ok();
        dir
    }

    #[test]
    fn test_open_and_list_tensors() {
        let path = test_dir().join("test_open_list.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        assert_eq!(rt.model_name(), "test-model");
        assert_eq!(rt.architecture(), "test");
        assert_eq!(rt.tensor_count(), 10);

        let names = rt.tensor_names();
        assert_eq!(names.len(), 10);
        assert!(names.contains(&"layer_0_weight"));
        assert!(names.contains(&"layer_9_weight"));
    }

    #[test]
    fn test_tensor_access() {
        let path = test_dir().join("test_tensor_access.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        // Access a specific tensor
        let data = rt.tensor("layer_5_weight").expect("Failed to get tensor");
        assert_eq!(data.len(), 64);
        assert_eq!(data[0], (5 * 64) as u8);
        assert_eq!(data[63], (5 * 64 + 63) as u8);
    }

    #[test]
    fn test_tensor_view_zero_copy() {
        let path = test_dir().join("test_view.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        // Zero-copy borrowed view — must point to valid data
        let view: &[u8] = rt
            .tensor_view("layer_0_weight")
            .expect("Failed to get tensor view");
        assert_eq!(view.len(), 64);
        assert_eq!(view[0], 0u8);
        assert_eq!(view[63], 63u8);
    }

    #[test]
    fn test_tensor_byte_view_zero_copy() {
        let path = test_dir().join("test_byte_view.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        // Zero-copy byte-range view
        let view: &[u8] = rt
            .tensor_byte_view("layer_0_weight", 16..32)
            .expect("Failed to get byte view");
        assert_eq!(view.len(), 16);
        assert_eq!(view[0], 16u8);
        assert_eq!(view[15], 31u8);
    }

    #[test]
    fn test_tensor_view_lifetime() {
        let path = test_dir().join("test_view_lifetime.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        // The view borrows from rt — the borrow checker enforces this.
        // This test verifies the view is valid while rt exists.
        let view = rt.tensor_view("layer_0_weight").unwrap();
        let first_byte = view[0];
        assert_eq!(first_byte, 0u8);
        // view is dropped before rt (or with rt) — safety guaranteed by Rust's borrow checker.
    }

    #[test]
    fn test_tensor_view_invalid_name() {
        let path = test_dir().join("test_view_invalid.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        let result = rt.tensor_view("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_view_invalid_range() {
        let path = test_dir().join("test_view_range.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        let result = rt.tensor_byte_view("layer_0_weight", 0..999);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_rows_2d() {
        use axon_core::AxonBuilder;
        use axon_core::DType;
        use std::fs;

        // Build a 2D tensor: 16 rows x 8 cols, U8
        let data: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let path = test_dir().join("test_rows_2d.axon");
        let axon = AxonBuilder::new()
            .add_tensor("mat", data.clone(), DType::U8, &[16, 8])
            .build()
            .unwrap();
        fs::write(&path, &axon).unwrap();

        let rt = AxonRuntime::open(&path).unwrap();

        // First 4 rows (0..4)
        let rows = rt.tensor_rows("mat", 0, 4).unwrap();
        assert_eq!(rows.len(), 4 * 8); // 4 rows x 8 cols
        assert_eq!(rows[0], 0);
        assert_eq!(rows[31], 31);

        // Middle rows (4..8)
        let rows = rt.tensor_rows("mat", 4, 8).unwrap();
        assert_eq!(rows.len(), 4 * 8);
        assert_eq!(rows[0], 32);
        assert_eq!(rows[31], 63);

        // Last rows (12..16)
        let rows = rt.tensor_rows("mat", 12, 16).unwrap();
        assert_eq!(rows.len(), 4 * 8);
        assert_eq!(rows[0], 96);
        assert_eq!(rows[31], 127);

        // Full tensor via rows
        let rows = rt.tensor_rows("mat", 0, 16).unwrap();
        let full = rt.tensor("mat").unwrap();
        assert_eq!(rows, &full[..]);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_tensor_rows_invalid_range() {
        use axon_core::AxonBuilder;
        use axon_core::DType;
        use std::fs;

        let data = vec![0u8; 64];
        let path = test_dir().join("test_rows_invalid.axon");
        let axon = AxonBuilder::new()
            .add_tensor("mat", data, DType::U8, &[8, 8])
            .build()
            .unwrap();
        fs::write(&path, &axon).unwrap();

        let rt = AxonRuntime::open(&path).unwrap();

        // start_row >= total_rows
        assert!(rt.tensor_rows("mat", 10, 12).is_err());

        // end_row > total_rows
        assert!(rt.tensor_rows("mat", 0, 20).is_err());

        // start_row > end_row
        assert!(rt.tensor_rows("mat", 5, 3).is_err());

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_tensor_rows_on_1d_fails() {
        use axon_core::AxonBuilder;
        use axon_core::DType;
        use std::fs;

        let data = vec![0u8; 64];
        let path = test_dir().join("test_rows_1d.axon");
        let axon = AxonBuilder::new()
            .add_tensor("vec", data, DType::U8, &[64])
            .build()
            .unwrap();
        fs::write(&path, &axon).unwrap();

        let rt = AxonRuntime::open(&path).unwrap();
        assert!(rt.tensor_rows("vec", 0, 10).is_err());

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_tensor_info() {
        let path = test_dir().join("test_tensor_info.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        let info = rt
            .tensor_info("layer_0_weight")
            .expect("Failed to get info");

        assert_eq!(info.name, "layer_0_weight");
        assert_eq!(info.dtype, DType::U8);
        assert_eq!(info.shape, vec![64]);
        assert_eq!(info.data_size, 64);
    }

    #[test]
    fn test_tensor_byte_range() {
        let path = test_dir().join("test_byte_range.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        // Load first 16 bytes of layer_3_weight
        let partial = rt
            .tensor_byte_range("layer_3_weight", 0, 16)
            .expect("Failed to get byte range");
        assert_eq!(partial.len(), 16);
        assert_eq!(partial[0], (3 * 64) as u8);
        assert_eq!(partial[15], (3 * 64 + 15) as u8);

        // Load bytes 32-47
        let mid = rt
            .tensor_byte_range("layer_3_weight", 32, 16)
            .expect("Failed to get mid range");
        assert_eq!(mid[0], (3 * 64 + 32) as u8);
    }

    #[test]
    fn test_tensor_not_found() {
        let path = test_dir().join("test_not_found.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        let result = rt.tensor("nonexistent_tensor");
        assert!(result.is_err());
        match result {
            Err(AxonError::TensorNotFound(name)) => assert_eq!(name, "nonexistent_tensor"),
            other => panic!("Expected TensorNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_byte_range_out_of_bounds() {
        let path = test_dir().join("test_range_oob.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        // Request beyond tensor size
        let result = rt.tensor_byte_range("layer_0_weight", 0, 999);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_count_accesses() {
        let path = test_dir().join("test_stats.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        assert_eq!(rt.stats().tensor_accesses(), 0);

        rt.tensor("layer_0_weight").ok();
        assert_eq!(rt.stats().tensor_accesses(), 1);

        rt.tensor("layer_1_weight").ok();
        assert_eq!(rt.stats().tensor_accesses(), 2);
    }

    #[test]
    fn test_tensors_list() {
        let path = test_dir().join("test_tensors_list.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");
        let all = rt.tensors();
        assert_eq!(all.len(), 10);

        // Verify info for each
        for info in &all {
            assert!(info.name.starts_with("layer_"));
            assert_eq!(info.data_size, 64);
        }
    }

    #[test]
    fn test_file_size_matches() {
        let path = test_dir().join("test_file_size.axon");
        build_test_axon(&path);

        let rt = AxonRuntime::open(&path).expect("Failed to open runtime");

        let file_meta = fs::metadata(&path).expect("Failed to get metadata");
        assert_eq!(rt.file_size(), file_meta.len());
    }

    #[test]
    fn test_open_nonexistent_fails() {
        let result = AxonRuntime::open("/tmp/axon_nonexistent_837465.axon");
        assert!(result.is_err());
    }
}
