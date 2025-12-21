//! Serialization for [`Value`] types.
//!
//! This module provides compact binary encoding for the [`Value`] enum,
//! which represents all possible property values in `ManifoldDB`.
//!
//! # Format
//!
//! Each value is encoded with a 1-byte type tag followed by the payload:
//!
//! - `Null`: `0x00`
//! - `Bool`: `0x01` + `0x00` (false) or `0x01` (true)
//! - `Int`: `0x02` + 8 bytes (big-endian i64)
//! - `Float`: `0x03` + 8 bytes (IEEE 754 f64)
//! - `String`: `0x04` + 4 bytes length + UTF-8 bytes
//! - `Bytes`: `0x05` + 4 bytes length + raw bytes
//! - `Vector`: `0x06` + 4 bytes length (count) + f32 values (little-endian)
//! - `Array`: `0x07` + 4 bytes length (count) + encoded values

use crate::error::CoreError;
use crate::types::Value;

use super::traits::{Decoder, Encoder};

/// Type tags for value variants.
mod tags {
    pub const NULL: u8 = 0x00;
    pub const BOOL: u8 = 0x01;
    pub const INT: u8 = 0x02;
    pub const FLOAT: u8 = 0x03;
    pub const STRING: u8 = 0x04;
    pub const BYTES: u8 = 0x05;
    pub const VECTOR: u8 = 0x06;
    pub const ARRAY: u8 = 0x07;
}

impl Encoder for Value {
    fn encode(&self) -> Result<Vec<u8>, CoreError> {
        let mut buf = Vec::new();
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    fn encode_to(&self, buf: &mut Vec<u8>) -> Result<(), CoreError> {
        match self {
            Self::Null => buf.push(tags::NULL),
            Self::Bool(b) => {
                buf.push(tags::BOOL);
                buf.push(u8::from(*b));
            }
            Self::Int(i) => {
                buf.push(tags::INT);
                buf.extend_from_slice(&i.to_be_bytes());
            }
            Self::Float(f) => {
                buf.push(tags::FLOAT);
                buf.extend_from_slice(&f.to_be_bytes());
            }
            Self::String(s) => {
                buf.push(tags::STRING);
                let bytes = s.as_bytes();
                let len = u32::try_from(bytes.len())
                    .map_err(|_| CoreError::Encoding("string too long".to_owned()))?;
                buf.extend_from_slice(&len.to_be_bytes());
                buf.extend_from_slice(bytes);
            }
            Self::Bytes(b) => {
                buf.push(tags::BYTES);
                let len = u32::try_from(b.len())
                    .map_err(|_| CoreError::Encoding("bytes too long".to_owned()))?;
                buf.extend_from_slice(&len.to_be_bytes());
                buf.extend_from_slice(b);
            }
            Self::Vector(v) => {
                buf.push(tags::VECTOR);
                let len = u32::try_from(v.len())
                    .map_err(|_| CoreError::Encoding("vector too long".to_owned()))?;
                buf.extend_from_slice(&len.to_be_bytes());
                // Use little-endian for f32 values for efficient memory copying
                for f in v {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            Self::Array(arr) => {
                buf.push(tags::ARRAY);
                let len = u32::try_from(arr.len())
                    .map_err(|_| CoreError::Encoding("array too long".to_owned()))?;
                buf.extend_from_slice(&len.to_be_bytes());
                for val in arr {
                    val.encode_to(buf)?;
                }
            }
        }
        Ok(())
    }
}

impl Decoder for Value {
    fn decode(bytes: &[u8]) -> Result<Self, CoreError> {
        let (value, _) = decode_value(bytes)?;
        Ok(value)
    }
}

/// Decode a value and return the number of bytes consumed.
///
/// This is useful for decoding arrays of values where we need to know
/// where each value ends.
pub fn decode_value(bytes: &[u8]) -> Result<(Value, usize), CoreError> {
    if bytes.is_empty() {
        return Err(CoreError::Encoding("unexpected end of input".to_owned()));
    }

    let tag = bytes[0];
    let rest = &bytes[1..];

    match tag {
        tags::NULL => Ok((Value::Null, 1)),
        tags::BOOL => {
            if rest.is_empty() {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            Ok((Value::Bool(rest[0] != 0), 2))
        }
        tags::INT => {
            if rest.len() < 8 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let bytes: [u8; 8] = rest[..8].try_into().map_err(|_| {
                CoreError::Encoding("failed to read i64 bytes".to_owned())
            })?;
            Ok((Value::Int(i64::from_be_bytes(bytes)), 9))
        }
        tags::FLOAT => {
            if rest.len() < 8 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let bytes: [u8; 8] = rest[..8].try_into().map_err(|_| {
                CoreError::Encoding("failed to read f64 bytes".to_owned())
            })?;
            Ok((Value::Float(f64::from_be_bytes(bytes)), 9))
        }
        tags::STRING => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4].try_into().map_err(|_| {
                CoreError::Encoding("failed to read length".to_owned())
            })?;
            let len = u32::from_be_bytes(len_bytes) as usize;
            if rest.len() < 4 + len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let s = String::from_utf8(rest[4..4 + len].to_vec())
                .map_err(|e| CoreError::Encoding(format!("invalid UTF-8: {e}")))?;
            Ok((Value::String(s), 5 + len))
        }
        tags::BYTES => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4].try_into().map_err(|_| {
                CoreError::Encoding("failed to read length".to_owned())
            })?;
            let len = u32::from_be_bytes(len_bytes) as usize;
            if rest.len() < 4 + len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            Ok((Value::Bytes(rest[4..4 + len].to_vec()), 5 + len))
        }
        tags::VECTOR => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4].try_into().map_err(|_| {
                CoreError::Encoding("failed to read length".to_owned())
            })?;
            let count = u32::from_be_bytes(len_bytes) as usize;
            let byte_len = count * 4;
            if rest.len() < 4 + byte_len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let mut vec = Vec::with_capacity(count);
            for i in 0..count {
                let offset = 4 + i * 4;
                let f_bytes: [u8; 4] = rest[offset..offset + 4].try_into().map_err(|_| {
                    CoreError::Encoding("failed to read f32 bytes".to_owned())
                })?;
                vec.push(f32::from_le_bytes(f_bytes));
            }
            Ok((Value::Vector(vec), 5 + byte_len))
        }
        tags::ARRAY => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4].try_into().map_err(|_| {
                CoreError::Encoding("failed to read length".to_owned())
            })?;
            let count = u32::from_be_bytes(len_bytes) as usize;
            let mut arr = Vec::with_capacity(count);
            let mut offset = 5; // tag + length bytes
            for _ in 0..count {
                let (val, consumed) = decode_value(&bytes[offset..])?;
                arr.push(val);
                offset += consumed;
            }
            Ok((Value::Array(arr), offset))
        }
        _ => Err(CoreError::Encoding(format!("unknown type tag: {tag:#x}"))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_null() {
        let original = Value::Null;
        let encoded = original.encode().unwrap();
        assert_eq!(encoded, vec![0x00]);
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_bool() {
        for b in [true, false] {
            let original = Value::Bool(b);
            let encoded = original.encode().unwrap();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn encode_decode_int() {
        for i in [0i64, 1, -1, i64::MIN, i64::MAX] {
            let original = Value::Int(i);
            let encoded = original.encode().unwrap();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn encode_decode_float() {
        for f in [0.0f64, 1.0, -1.0, f64::MIN, f64::MAX, f64::INFINITY, f64::NEG_INFINITY] {
            let original = Value::Float(f);
            let encoded = original.encode().unwrap();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn encode_decode_string() {
        for s in ["", "hello", "hello world", "\u{1F600}"] {
            let original = Value::String(s.to_owned());
            let encoded = original.encode().unwrap();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn encode_decode_bytes() {
        for b in [vec![], vec![0u8], vec![1, 2, 3, 4, 5]] {
            let original = Value::Bytes(b);
            let encoded = original.encode().unwrap();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn encode_decode_vector() {
        for v in [vec![], vec![0.0f32], vec![0.1, 0.2, 0.3]] {
            let original = Value::Vector(v);
            let encoded = original.encode().unwrap();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn encode_decode_array() {
        let original = Value::Array(vec![
            Value::Null,
            Value::Bool(true),
            Value::Int(42),
            Value::String("nested".to_owned()),
        ]);
        let encoded = original.encode().unwrap();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_nested_array() {
        let original = Value::Array(vec![
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
            Value::Array(vec![Value::String("a".to_owned()), Value::String("b".to_owned())]),
        ]);
        let encoded = original.encode().unwrap();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_invalid_tag() {
        let bytes = [0xFF];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_empty_input() {
        let result = Value::decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_int() {
        let bytes = [tags::INT, 0, 0, 0]; // Only 4 bytes instead of 8
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }
}
