//! # MmapStore
//!
//! Safe, owned wrapper around a read-only memory-mapped file.
//!
//! The store owns the `Mmap` handle. It provides byte-range access and
//! never reads beyond the file boundaries. No tensor data is copied into
//! application heap until `read_bytes` is called — the OS lazily pages
//! in the requested ranges.
//!
//! ## Safety
//!
//! `read_bytes` returns an owned `Vec<u8>`. This is safe regardless of
//! how long the caller holds the result — the mmap handle stays alive as
//! long as the `MmapStore` lives, and the data is copied.
//!
//! `raw_slice` returns a borrowed `&[u8]` and must not outlive the store.
//! It is crate-internal only.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use axon_core::{AxonError, AxonResult};

/// Owns a read-only memory-mapped file.
///
/// The entire file is mapped with `MAP_PRIVATE | PROT_READ`. The OS manages
/// which pages are resident in RAM — only the byte ranges you actually read
/// are faulted in from disk.
pub struct MmapStore {
    mmap: Mmap,
    len: u64,
}

impl MmapStore {
    /// Open a file and memory-map the entire contents.
    pub fn open<P: AsRef<Path>>(path: P) -> AxonResult<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            log::error!("Failed to open file: {}", e);
            AxonError::Io(e)
        })?;

        let metadata = file.metadata().map_err(|e| {
            log::error!("Failed to read file metadata: {}", e);
            AxonError::Io(e)
        })?;
        let len = metadata.len();

        // SAFETY: The file is opened read-only. MAP_PRIVATE means any
        // accidental writes are copy-on-write and never persisted.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                log::error!("Failed to mmap file: {}", e);
                AxonError::Mmap(e.to_string())
            })?
        };

        log::debug!("Mmap'd {} ({} bytes)", path.as_ref().display(), len);

        Ok(Self { mmap, len })
    }

    /// Read a byte range from the mmap into an owned `Vec<u8>`.
    ///
    /// This is the primary public access method. It copies bytes from the
    /// mmap into heap memory, which means the caller can hold the result
    /// independently of this store's lifetime.
    ///
    /// The first call to a given byte range triggers a page fault — the OS
    /// loads the corresponding file pages from disk into the page cache.
    /// Subsequent calls to the same range hit the page cache (RAM).
    pub fn read_bytes(&self, offset: u64, size: u64) -> AxonResult<Vec<u8>> {
        self.validate_range(offset, size)?;
        let start = offset as usize;
        let end = start + size as usize;
        Ok(self.mmap[start..end].to_vec())
    }

    /// Get a raw byte slice into the mmap.
    ///
    /// This is zero-copy but the returned slice borrows from the mmap.
    /// It must not outlive this store. Crate-internal only.
    pub(crate) fn raw_slice(&self, offset: u64, size: u64) -> Option<&[u8]> {
        self.validate_range(offset, size).ok()?;
        let start = offset as usize;
        let end = start + size as usize;
        Some(&self.mmap[start..end])
    }

    /// The total mapped file size in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the store is empty (zero-length file).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Validate that `[offset, offset + size)` is within the mapped range.
    fn validate_range(&self, offset: u64, size: u64) -> AxonResult<()> {
        if offset > self.len {
            return Err(AxonError::AlignmentError {
                offset,
                alignment: self.len,
            });
        }
        let end = offset.checked_add(size).ok_or_else(|| {
            let msg = format!("byte range {} + {} overflows u64", offset, size);
            AxonError::Mmap(msg)
        })?;
        if end > self.len {
            return Err(AxonError::UnexpectedEof {
                needed: end,
                available: self.len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_file(contents: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(contents).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_open_and_read() {
        let data = b"hello axon runtime";
        let f = create_temp_file(data);
        let store = MmapStore::open(f.path()).unwrap();
        assert!(!store.is_empty());

        let bytes = store.read_bytes(0, data.len() as u64).unwrap();
        assert_eq!(&bytes, data);
    }

    #[test]
    fn test_read_subrange() {
        let f = create_temp_file(b"abcdefghijklmnop");
        let store = MmapStore::open(f.path()).unwrap();

        let bytes = store.read_bytes(3, 5).unwrap();
        assert_eq!(&bytes, b"defgh");
    }

    #[test]
    fn test_read_beyond_eof_fails() {
        let f = create_temp_file(b"short");
        let store = MmapStore::open(f.path()).unwrap();

        let result = store.read_bytes(0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_open_nonexistent_file_fails() {
        let result = MmapStore::open("/tmp/axon_nonexistent_98273492.axon");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let f = create_temp_file(b"");
        let store = MmapStore::open(f.path()).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        let result = store.read_bytes(0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_raw_slice() {
        let f = create_temp_file(b"raw slice test");
        let store = MmapStore::open(f.path()).unwrap();

        let slice = store.raw_slice(4, 5).unwrap();
        assert_eq!(slice, b"slice");
    }

    #[test]
    fn test_raw_slice_out_of_bounds() {
        let f = create_temp_file(b"small");
        let store = MmapStore::open(f.path()).unwrap();

        assert!(store.raw_slice(0, 100).is_none());
        assert!(store.raw_slice(100, 1).is_none());
    }
}
