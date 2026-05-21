use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::error::{AxonError, AxonResult};
use crate::{AXON_MAGIC, AXON_VERSION, CACHE_LINE_SIZE};

/// The `AxonHeader` occupies the first 64 bytes of every .axon file.
/// Designed for zero-copy casting from an mmap'd region on 64-bit systems.
///
/// Layout (64 bytes total, aligned to 64):
///
/// | Offset | Size | Field              | Description                              |
/// |--------|------|--------------------|------------------------------------------|
/// | 0      | 4    | magic              | Magic bytes: "AXON"                      |
/// | 4      | 4    | version            | Format version (currently 1)             |
/// | 8      | 8    | manifest_offset    | Byte offset to the manifest block        |
/// | 16     | 8    | manifest_size      | Size of the manifest in bytes            |
/// | 24     | 8    | tensor_count       | Number of tensors in the file            |
/// | 32     | 8    | payload_offset     | Byte offset to the first tensor payload  |
/// | 40     | 8    | payload_size       | Total size of all tensor payloads        |
/// | 48     | 8    | checksum           | XXH3 checksum of header (bytes 0..48)    |
/// | 56     | 8    | flags              | Bit flags for format features            |
///
/// Total: 64 bytes = 1 cache line.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AxonHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub tensor_count: u64,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub checksum: u64,
    pub flags: u64,
}

impl Default for AxonHeader {
    fn default() -> Self {
        Self {
            magic: *AXON_MAGIC,
            version: AXON_VERSION,
            manifest_offset: HOT_START_SIZE,
            manifest_size: 0,
            tensor_count: 0,
            payload_offset: 0,
            payload_size: 0,
            checksum: 0,
            flags: AxonFlags::HAS_CHECKSUMS.bits(),
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct AxonFlags: u64 {
        const NONE          = 0;
        const HAS_CHECKSUMS = 1 << 0;
        const SHARDED       = 1 << 1;
        const INTERLEAVED   = 1 << 2;
        const COMPRESSED    = 1 << 3;
        const ENCRYPTED     = 1 << 4;
    }
}

const HOT_START_SIZE: u64 = 4096;

impl AxonHeader {
    pub const HEADER_SIZE: usize = 64;

    pub fn align_up(offset: u64, alignment: u64) -> u64 {
        (offset + alignment - 1) & !(alignment - 1)
    }

    pub fn new(tensor_count: u64, manifest_size: u64, payload_size: u64) -> Self {
        let manifest_offset = HOT_START_SIZE;
        let payload_offset = Self::align_up(manifest_offset + manifest_size, CACHE_LINE_SIZE);

        Self {
            magic: *AXON_MAGIC,
            version: AXON_VERSION,
            manifest_offset,
            manifest_size,
            tensor_count,
            payload_offset,
            payload_size,
            checksum: 0,
            flags: AxonFlags::HAS_CHECKSUMS.bits(),
        }
    }

    pub fn validate(&self) -> AxonResult<()> {
        if &self.magic != AXON_MAGIC {
            return Err(AxonError::InvalidMagic(self.magic));
        }
        if self.version != AXON_VERSION {
            return Err(AxonError::UnsupportedVersion(self.version));
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> AxonResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(Self::HEADER_SIZE);
        buf.write_all(&self.magic)?;
        buf.write_u32::<LittleEndian>(self.version)?;
        buf.write_u64::<LittleEndian>(self.manifest_offset)?;
        buf.write_u64::<LittleEndian>(self.manifest_size)?;
        buf.write_u64::<LittleEndian>(self.tensor_count)?;
        buf.write_u64::<LittleEndian>(self.payload_offset)?;
        buf.write_u64::<LittleEndian>(self.payload_size)?;
        buf.write_u64::<LittleEndian>(self.checksum)?;
        buf.write_u64::<LittleEndian>(self.flags)?;
        while buf.len() < Self::HEADER_SIZE {
            buf.write_u8(0)?;
        }
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> AxonResult<Self> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(AxonError::UnexpectedEof {
                needed: Self::HEADER_SIZE as u64,
                available: bytes.len() as u64,
            });
        }
        let mut cursor = Cursor::new(bytes);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        let version = cursor.read_u32::<LittleEndian>()?;
        let manifest_offset = cursor.read_u64::<LittleEndian>()?;
        let manifest_size = cursor.read_u64::<LittleEndian>()?;
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let payload_offset = cursor.read_u64::<LittleEndian>()?;
        let payload_size = cursor.read_u64::<LittleEndian>()?;
        let checksum = cursor.read_u64::<LittleEndian>()?;
        let flags = cursor.read_u64::<LittleEndian>()?;
        let header = Self {
            magic,
            version,
            manifest_offset,
            manifest_size,
            tensor_count,
            payload_offset,
            payload_size,
            checksum,
            flags,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn has_checksums(&self) -> bool {
        self.flags & AxonFlags::HAS_CHECKSUMS.bits() != 0
    }

    pub fn is_sharded(&self) -> bool {
        self.flags & AxonFlags::SHARDED.bits() != 0
    }
}
