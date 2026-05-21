use thiserror::Error;

#[derive(Error, Debug)]
pub enum AxonError {
    #[error("Invalid magic bytes: expected AXON, got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),

    #[error("Unexpected EOF: needed {needed} bytes, got {available}")]
    UnexpectedEof { needed: u64, available: u64 },

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Alignment error: offset {offset} is not aligned to {alignment} bytes")]
    AlignmentError { offset: u64, alignment: u64 },

    #[error("Checksum mismatch for tensor `{name}`: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Invalid dtype code: {0}")]
    InvalidDtype(u32),

    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),

    #[error("Mmap error: {0}")]
    Mmap(String),
}

pub type AxonResult<T> = Result<T, AxonError>;
