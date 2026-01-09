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
//! - `SparseVector`: `0x08` + 4 bytes length (count) + (u32 index, f32 value) pairs
//! - `MultiVector`: `0x09` + 4 bytes (vector count) + 4 bytes (dim) + f32 values (little-endian)

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
    pub const SPARSE_VECTOR: u8 = 0x08;
    pub const MULTI_VECTOR: u8 = 0x09;
    pub const POINT: u8 = 0x0A;
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
            Self::SparseVector(v) => {
                buf.push(tags::SPARSE_VECTOR);
                let len = u32::try_from(v.len())
                    .map_err(|_| CoreError::Encoding("sparse vector too long".to_owned()))?;
                buf.extend_from_slice(&len.to_be_bytes());
                // Encode each (index, value) pair
                for (idx, val) in v {
                    buf.extend_from_slice(&idx.to_le_bytes());
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            Self::MultiVector(vecs) => {
                buf.push(tags::MULTI_VECTOR);
                let count = u32::try_from(vecs.len())
                    .map_err(|_| CoreError::Encoding("multi-vector too long".to_owned()))?;
                buf.extend_from_slice(&count.to_be_bytes());
                // For variable-length multi-vectors, we store each vector with its length
                // Format: count + for each vector: (len + f32 values)
                for vec in vecs {
                    let vec_len = u32::try_from(vec.len()).map_err(|_| {
                        CoreError::Encoding("vector in multi-vector too long".to_owned())
                    })?;
                    buf.extend_from_slice(&vec_len.to_be_bytes());
                    for f in vec {
                        buf.extend_from_slice(&f.to_le_bytes());
                    }
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
            Self::Point { x, y, z, srid } => {
                buf.push(tags::POINT);
                // Encode SRID (4 bytes)
                buf.extend_from_slice(&srid.to_be_bytes());
                // Encode x (8 bytes)
                buf.extend_from_slice(&x.to_be_bytes());
                // Encode y (8 bytes)
                buf.extend_from_slice(&y.to_be_bytes());
                // Encode z presence flag (1 byte) and optional z (8 bytes)
                match z {
                    Some(z_val) => {
                        buf.push(1);
                        buf.extend_from_slice(&z_val.to_be_bytes());
                    }
                    None => {
                        buf.push(0);
                    }
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
            let bytes: [u8; 8] = rest[..8]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read i64 bytes".to_owned()))?;
            Ok((Value::Int(i64::from_be_bytes(bytes)), 9))
        }
        tags::FLOAT => {
            if rest.len() < 8 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let bytes: [u8; 8] = rest[..8]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read f64 bytes".to_owned()))?;
            Ok((Value::Float(f64::from_be_bytes(bytes)), 9))
        }
        tags::STRING => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read length".to_owned()))?;
            let len = usize::try_from(u32::from_be_bytes(len_bytes)).map_err(|_| {
                CoreError::Encoding("string length exceeds platform capacity".to_owned())
            })?;
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
            let len_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read length".to_owned()))?;
            let len = usize::try_from(u32::from_be_bytes(len_bytes)).map_err(|_| {
                CoreError::Encoding("bytes length exceeds platform capacity".to_owned())
            })?;
            if rest.len() < 4 + len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            Ok((Value::Bytes(rest[4..4 + len].to_vec()), 5 + len))
        }
        tags::VECTOR => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read length".to_owned()))?;
            let count = usize::try_from(u32::from_be_bytes(len_bytes)).map_err(|_| {
                CoreError::Encoding("vector count exceeds platform capacity".to_owned())
            })?;
            let byte_len = count
                .checked_mul(4)
                .ok_or_else(|| CoreError::Encoding("vector byte length overflow".to_owned()))?;
            if rest.len() < 4 + byte_len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let mut vec = Vec::with_capacity(count);
            for i in 0..count {
                let offset = 4 + i * 4;
                let f_bytes: [u8; 4] = rest[offset..offset + 4]
                    .try_into()
                    .map_err(|_| CoreError::Encoding("failed to read f32 bytes".to_owned()))?;
                vec.push(f32::from_le_bytes(f_bytes));
            }
            Ok((Value::Vector(vec), 5 + byte_len))
        }
        tags::SPARSE_VECTOR => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read length".to_owned()))?;
            let count = usize::try_from(u32::from_be_bytes(len_bytes)).map_err(|_| {
                CoreError::Encoding("sparse vector count exceeds platform capacity".to_owned())
            })?;
            // Each entry is 4 bytes (u32 index) + 4 bytes (f32 value) = 8 bytes
            let byte_len = count.checked_mul(8).ok_or_else(|| {
                CoreError::Encoding("sparse vector byte length overflow".to_owned())
            })?;
            if rest.len() < 4 + byte_len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let mut vec = Vec::with_capacity(count);
            for i in 0..count {
                let offset = 4 + i * 8;
                let idx_bytes: [u8; 4] = rest[offset..offset + 4]
                    .try_into()
                    .map_err(|_| CoreError::Encoding("failed to read u32 index".to_owned()))?;
                let val_bytes: [u8; 4] = rest[offset + 4..offset + 8]
                    .try_into()
                    .map_err(|_| CoreError::Encoding("failed to read f32 value".to_owned()))?;
                vec.push((u32::from_le_bytes(idx_bytes), f32::from_le_bytes(val_bytes)));
            }
            Ok((Value::SparseVector(vec), 5 + byte_len))
        }
        tags::MULTI_VECTOR => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let count_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read count".to_owned()))?;
            let count = usize::try_from(u32::from_be_bytes(count_bytes)).map_err(|_| {
                CoreError::Encoding("multi-vector count exceeds platform capacity".to_owned())
            })?;

            let mut vecs = Vec::with_capacity(count);
            let mut pos = 4; // Position after count

            for _ in 0..count {
                if rest.len() < pos + 4 {
                    return Err(CoreError::Encoding("unexpected end of input".to_owned()));
                }
                let vec_len_bytes: [u8; 4] = rest[pos..pos + 4]
                    .try_into()
                    .map_err(|_| CoreError::Encoding("failed to read vector length".to_owned()))?;
                let vec_len = usize::try_from(u32::from_be_bytes(vec_len_bytes)).map_err(|_| {
                    CoreError::Encoding("vector length exceeds platform capacity".to_owned())
                })?;
                pos += 4;

                let byte_len = vec_len
                    .checked_mul(4)
                    .ok_or_else(|| CoreError::Encoding("vector byte length overflow".to_owned()))?;
                if rest.len() < pos + byte_len {
                    return Err(CoreError::Encoding("unexpected end of input".to_owned()));
                }

                let mut vec = Vec::with_capacity(vec_len);
                for i in 0..vec_len {
                    let offset = pos + i * 4;
                    let f_bytes: [u8; 4] = rest[offset..offset + 4]
                        .try_into()
                        .map_err(|_| CoreError::Encoding("failed to read f32 bytes".to_owned()))?;
                    vec.push(f32::from_le_bytes(f_bytes));
                }
                vecs.push(vec);
                pos += byte_len;
            }
            Ok((Value::MultiVector(vecs), 1 + pos)) // 1 for tag
        }
        tags::ARRAY => {
            if rest.len() < 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read length".to_owned()))?;
            let count = usize::try_from(u32::from_be_bytes(len_bytes)).map_err(|_| {
                CoreError::Encoding("array count exceeds platform capacity".to_owned())
            })?;
            let mut arr = Vec::with_capacity(count);
            let mut offset = 5; // tag + length bytes
            for _ in 0..count {
                let (val, consumed) = decode_value(&bytes[offset..])?;
                arr.push(val);
                offset += consumed;
            }
            Ok((Value::Array(arr), offset))
        }
        tags::POINT => {
            // Point format: tag(1) + srid(4) + x(8) + y(8) + z_flag(1) + optional z(8)
            // Minimum size: 1 + 4 + 8 + 8 + 1 = 22 bytes (without z)
            // Maximum size: 1 + 4 + 8 + 8 + 1 + 8 = 30 bytes (with z)
            if rest.len() < 21 {
                // 4 (srid) + 8 (x) + 8 (y) + 1 (z_flag)
                return Err(CoreError::Encoding("unexpected end of input for point".to_owned()));
            }
            let srid_bytes: [u8; 4] = rest[..4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read srid".to_owned()))?;
            let srid = u32::from_be_bytes(srid_bytes);

            let x_bytes: [u8; 8] = rest[4..12]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read x".to_owned()))?;
            let x = f64::from_be_bytes(x_bytes);

            let y_bytes: [u8; 8] = rest[12..20]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read y".to_owned()))?;
            let y = f64::from_be_bytes(y_bytes);

            let z_flag = rest[20];
            let (z, consumed) = if z_flag != 0 {
                if rest.len() < 29 {
                    // 21 + 8 for z
                    return Err(CoreError::Encoding(
                        "unexpected end of input for point z coordinate".to_owned(),
                    ));
                }
                let z_bytes: [u8; 8] = rest[21..29]
                    .try_into()
                    .map_err(|_| CoreError::Encoding("failed to read z".to_owned()))?;
                (Some(f64::from_be_bytes(z_bytes)), 30) // tag(1) + srid(4) + x(8) + y(8) + z_flag(1) + z(8)
            } else {
                (None, 22) // tag(1) + srid(4) + x(8) + y(8) + z_flag(1)
            };

            Ok((Value::Point { x, y, z, srid }, consumed))
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
    fn encode_decode_sparse_vector() {
        for v in [vec![], vec![(0u32, 0.5f32)], vec![(0, 0.1), (10, 0.2), (100, 0.3), (1000, 0.4)]]
        {
            let original = Value::SparseVector(v);
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

    #[test]
    fn decode_truncated_float() {
        let bytes = [tags::FLOAT, 0, 0, 0, 0]; // Only 5 bytes instead of 9
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_string_length() {
        let bytes = [tags::STRING, 0, 0]; // Only 2 bytes for length instead of 4
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_string_data() {
        // String with length 10 but only 3 bytes of data
        let bytes = [tags::STRING, 0, 0, 0, 10, b'a', b'b', b'c'];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_invalid_utf8_string() {
        // String with invalid UTF-8 sequence
        let bytes = [tags::STRING, 0, 0, 0, 2, 0xFF, 0xFE];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_bytes_data() {
        // Bytes with length 10 but only 3 bytes of data
        let bytes = [tags::BYTES, 0, 0, 0, 10, 1, 2, 3];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_vector() {
        // Vector with count 5 but only 1 f32 worth of data
        let bytes = [tags::VECTOR, 0, 0, 0, 5, 0, 0, 0, 0];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_sparse_vector() {
        // SparseVector with count 2 but only 1 entry
        let bytes = [
            tags::SPARSE_VECTOR,
            0,
            0,
            0,
            2, // count = 2
            0,
            0,
            0,
            1, // index = 1
            0,
            0,
            0,
            0, // value = 0.0
        ];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_multi_vector_count() {
        // MultiVector with only 2 bytes for count
        let bytes = [tags::MULTI_VECTOR, 0, 0];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_multi_vector_inner() {
        // MultiVector with count 1 but missing inner vector length
        let bytes = [tags::MULTI_VECTOR, 0, 0, 0, 1];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_array_element() {
        // Array with count 2 but only 1 element
        let bytes = [tags::ARRAY, 0, 0, 0, 2, tags::NULL];
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_bool() {
        let bytes = [tags::BOOL]; // Missing boolean value byte
        let result = Value::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn encode_decode_multi_vector() {
        // Empty multi-vector
        let original = Value::MultiVector(vec![]);
        let encoded = original.encode().unwrap();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);

        // Single vector
        let original = Value::MultiVector(vec![vec![0.1, 0.2, 0.3]]);
        let encoded = original.encode().unwrap();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);

        // Multiple vectors (ColBERT-style token embeddings)
        let original =
            Value::MultiVector(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6], vec![0.7, 0.8, 0.9]]);
        let encoded = original.encode().unwrap();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);

        // Variable-length vectors
        let original =
            Value::MultiVector(vec![vec![0.1, 0.2], vec![0.3, 0.4, 0.5, 0.6], vec![0.7]]);
        let encoded = original.encode().unwrap();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }
}
