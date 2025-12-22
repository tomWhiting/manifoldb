//! WAL error types and result aliases

use std::io;

/// Result type alias for WAL operations
pub type WalResult<T> = Result<T, WalError>;

/// Errors that can occur during WAL operations
#[derive(Debug, thiserror::Error)]
pub enum WalError {
    /// I/O error during file operations
    #[error("WAL I/O error: {0}")]
    Io(#[from] io::Error),

    /// Checksum mismatch - data corruption detected
    #[error("WAL checksum mismatch at LSN {lsn}: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch {
        /// The LSN of the corrupted entry
        lsn: u64,
        /// Expected checksum value
        expected: u32,
        /// Actual computed checksum
        actual: u32,
    },

    /// Entry deserialization failed
    #[error("WAL entry deserialization failed: {0}")]
    Deserialize(String),

    /// Entry serialization failed
    #[error("WAL entry serialization failed: {0}")]
    Serialize(String),

    /// Invalid WAL file format or magic number
    #[error("Invalid WAL file format: {0}")]
    InvalidFormat(String),

    /// WAL file is truncated or incomplete
    #[error("WAL file truncated at offset {offset}")]
    Truncated {
        /// Byte offset where truncation was detected
        offset: u64,
    },

    /// Attempted to write with an LSN that's not monotonically increasing
    #[error("LSN {attempted} is not greater than current LSN {current}")]
    InvalidLsn {
        /// The LSN that was attempted
        attempted: u64,
        /// The current highest LSN
        current: u64,
    },

    /// Checkpoint failed
    #[error("Checkpoint failed: {0}")]
    Checkpoint(String),

    /// WAL file is locked by another process
    #[error("WAL file is locked by another process")]
    Locked,

    /// Recovery encountered unrecoverable state
    #[error("WAL recovery failed: {0}")]
    Recovery(String),
}

impl WalError {
    /// Returns true if this error indicates data corruption
    pub const fn is_corruption(&self) -> bool {
        matches!(
            self,
            Self::ChecksumMismatch { .. } | Self::Truncated { .. } | Self::InvalidFormat(_)
        )
    }

    /// Returns true if this error is recoverable (can retry)
    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Self::Io(_) | Self::Locked)
    }
}
