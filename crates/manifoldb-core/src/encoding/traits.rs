//! Encoding and decoding traits for serialization.

use crate::CoreError;

/// A trait for types that can be encoded to bytes.
///
/// This trait provides a unified interface for serializing types to a binary format
/// suitable for storage. Implementations should be efficient and produce compact output.
pub trait Encoder: Sized {
    /// Encode this value to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails (e.g., due to invalid data).
    fn encode(&self) -> Result<Vec<u8>, CoreError>;

    /// Encode this value into a pre-allocated buffer.
    ///
    /// This method appends the encoded bytes to the provided buffer,
    /// which can be more efficient when encoding multiple values.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    fn encode_to(&self, buf: &mut Vec<u8>) -> Result<(), CoreError>;
}

/// A trait for types that can be decoded from bytes.
///
/// This trait provides a unified interface for deserializing types from a binary format.
pub trait Decoder: Sized {
    /// Decode a value from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails (e.g., invalid data, truncated input).
    fn decode(bytes: &[u8]) -> Result<Self, CoreError>;
}

/// Format version for serialized data.
///
/// This version number is embedded in serialized data to support
/// forward-compatible schema evolution.
pub const FORMAT_VERSION: u8 = 1;
