use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::error::{AxonError, AxonResult};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    I64 = 4,
    U8 = 5,
    Q4 = 6,
    Q8 = 7,
    F8E4M3 = 8,
    F8E5M2 = 9,
    I8 = 10,
    I16 = 11,
}

impl DType {
    pub fn from_code(code: u32) -> AxonResult<Self> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::BF16),
            3 => Ok(Self::I32),
            4 => Ok(Self::I64),
            5 => Ok(Self::U8),
            6 => Ok(Self::Q4),
            7 => Ok(Self::Q8),
            8 => Ok(Self::F8E4M3),
            9 => Ok(Self::F8E5M2),
            10 => Ok(Self::I8),
            11 => Ok(Self::I16),
            _ => Err(AxonError::InvalidDtype(code)),
        }
    }

    pub fn to_code(self) -> u32 {
        self as u32
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::Q4 => 1,
            Self::Q8 => 1,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::I8 => 1,
            Self::I16 => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "FP32",
            Self::F16 => "FP16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::Q4 => "Q4",
            Self::Q8 => "Q8",
            Self::F8E4M3 => "FP8_E4M3",
            Self::F8E5M2 => "FP8_E5M2",
            Self::I8 => "I8",
            Self::I16 => "I16",
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Affinity {
    Default = 0,
    Hbm = 1,
    SystemRam = 2,
    Llc = 3,
}

impl Affinity {
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::Hbm,
            2 => Self::SystemRam,
            3 => Self::Llc,
            _ => Self::Default,
        }
    }
}

pub const MAX_TENSOR_RANK: usize = 8;
pub const TENSOR_NAME_MAX: usize = 64;

/// Fixed-size (192-byte) descriptor for each tensor in an .axon file.
///
/// Layout (192 bytes, aligned to 64):
///
/// | Offset | Size | Field       | Description                      |
/// |--------|------|-------------|----------------------------------|
/// | 0      | 64   | name        | Null-terminated tensor name      |
/// | 64     | 4    | dtype       | Data type code                   |
/// | 68     | 4    | rank        | Number of dimensions             |
/// | 72     | 64   | shape       | Dimension sizes (8 x u64)        |
/// | 136    | 8    | data_offset | Byte offset to raw tensor data   |
/// | 144    | 8    | data_size   | Size of tensor data in bytes     |
/// | 152    | 4    | affinity    | Memory affinity hint             |
/// | 156    | 4    | padding     | Reserved                         |
/// | 160    | 8    | checksum    | XXH3 checksum of tensor data     |
/// | 168    | 24   | padding2    | Padding to 192 bytes             |
///
/// Total: 192 bytes.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: [u8; TENSOR_NAME_MAX],
    pub dtype: u32,
    pub rank: u32,
    pub shape: [u64; MAX_TENSOR_RANK],
    pub data_offset: u64,
    pub data_size: u64,
    pub affinity: u32,
    pub _padding: u32,
    pub checksum: u64,
}

impl TensorDescriptor {
    pub const SIZE: usize = 192;

    pub fn new(
        name: &str,
        dtype: DType,
        shape: &[u64],
        data_offset: u64,
        data_size: u64,
        affinity: Affinity,
        checksum: u64,
    ) -> Self {
        let mut name_bytes = [0u8; TENSOR_NAME_MAX];
        let name_slice = name.as_bytes();
        let len = name_slice.len().min(TENSOR_NAME_MAX - 1);
        name_bytes[..len].copy_from_slice(&name_slice[..len]);
        name_bytes[len] = 0;

        let mut shape_arr = [0u64; MAX_TENSOR_RANK];
        for (i, &d) in shape.iter().enumerate().take(MAX_TENSOR_RANK) {
            shape_arr[i] = d;
        }

        Self {
            name: name_bytes,
            dtype: dtype.to_code(),
            rank: shape.len() as u32,
            shape: shape_arr,
            data_offset,
            data_size,
            affinity: affinity as u32,
            _padding: 0,
            checksum,
        }
    }

    pub fn name_str(&self) -> &str {
        let end = self
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(TENSOR_NAME_MAX);
        std::str::from_utf8(&self.name[..end]).unwrap_or("")
    }

    pub fn shape_vec(&self) -> Vec<u64> {
        self.shape[..self.rank as usize].to_vec()
    }

    pub fn num_elements(&self) -> u64 {
        self.shape[..self.rank as usize].iter().product()
    }

    pub fn dtype(&self) -> AxonResult<DType> {
        DType::from_code(self.dtype)
    }
    pub fn affinity(&self) -> Affinity {
        Affinity::from_code(self.affinity)
    }

    pub fn to_bytes(&self) -> AxonResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(Self::SIZE);
        buf.write_all(&self.name)?;
        buf.write_u32::<LittleEndian>(self.dtype)?;
        buf.write_u32::<LittleEndian>(self.rank)?;
        for &s in &self.shape {
            buf.write_u64::<LittleEndian>(s)?;
        }
        buf.write_u64::<LittleEndian>(self.data_offset)?;
        buf.write_u64::<LittleEndian>(self.data_size)?;
        buf.write_u32::<LittleEndian>(self.affinity)?;
        buf.write_u32::<LittleEndian>(self._padding)?;
        buf.write_u64::<LittleEndian>(self.checksum)?;
        while buf.len() < Self::SIZE {
            buf.write_u8(0)?;
        }
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> AxonResult<Self> {
        if bytes.len() < Self::SIZE {
            return Err(AxonError::UnexpectedEof {
                needed: Self::SIZE as u64,
                available: bytes.len() as u64,
            });
        }
        let mut cursor = Cursor::new(bytes);
        let mut name = [0u8; TENSOR_NAME_MAX];
        cursor.read_exact(&mut name)?;
        let dtype = cursor.read_u32::<LittleEndian>()?;
        let rank = cursor.read_u32::<LittleEndian>()?;
        let mut shape = [0u64; MAX_TENSOR_RANK];
        for s in &mut shape {
            *s = cursor.read_u64::<LittleEndian>()?;
        }
        let data_offset = cursor.read_u64::<LittleEndian>()?;
        let data_size = cursor.read_u64::<LittleEndian>()?;
        let affinity = cursor.read_u32::<LittleEndian>()?;
        let _padding = cursor.read_u32::<LittleEndian>()?;
        let checksum = cursor.read_u64::<LittleEndian>()?;
        Ok(Self {
            name,
            dtype,
            rank,
            shape,
            data_offset,
            data_size,
            affinity,
            _padding,
            checksum,
        })
    }
}
