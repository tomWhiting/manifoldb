//! Sort-order preserving encoding for property values.
//!
//! This module provides a binary encoding for [`Value`] types that preserves
//! lexicographic sort order when comparing the encoded bytes. This is essential
//! for secondary indexes where we need efficient range scans.
//!
//! # Encoding Design
//!
//! The encoding ensures that comparing encoded bytes produces the same ordering
//! as comparing the original values. Key design choices:
//!
//! ## Type Ordering
//!
//! Different types are ordered by their type tag (sorted from lowest to highest):
//! - `Null` (0x00) - sorts first
//! - `Bool` (0x01) - false before true
//! - `Int` (0x02) - negative to positive
//! - `Float` (0x03) - negative to positive, NaN handling
//! - `String` (0x04) - lexicographic UTF-8 order
//! - `Bytes` (0x05) - lexicographic byte order
//!
//! Vectors, arrays, sparse vectors, and multi-vectors are not supported for
//! indexing as they don't have a natural total ordering.
//!
//! ## Integer Encoding
//!
//! Integers use a "sign-flip" encoding:
//! - XOR with `0x8000_0000_0000_0000` flips the sign bit
//! - This makes negative numbers sort before positive numbers
//! - Result is stored in big-endian format
//!
//! ## Float Encoding
//!
//! Floats use IEEE 754 bit representation with transformations:
//! - Positive floats: flip sign bit (XOR with `0x8000_0000_0000_0000`)
//! - Negative floats: flip all bits (XOR with `0xFFFF_FFFF_FFFF_FFFF`)
//! - This preserves the natural numeric ordering
//! - NaN is encoded to sort after all other values
//!
//! ## String and Bytes Encoding
//!
//! Strings and bytes use null-terminated encoding with escape sequences:
//! - `0x00` in the data is escaped to `0x00 0x01`
//! - The sequence ends with `0x00 0x00` (double null terminator)
//!
//! This preserves lexicographic ordering:
//! - `"a" < "aa" < "ab" < "b"`
//!
//! And enables efficient prefix scans for `LIKE 'prefix%'` queries.
//!
//! # Example
//!
//! ```
//! use manifoldb_core::encoding::sortable::{encode_sortable, decode_sortable};
//! use manifoldb_core::types::Value;
//!
//! let values = vec![
//!     Value::Int(-10),
//!     Value::Int(0),
//!     Value::Int(10),
//! ];
//!
//! let mut encoded: Vec<_> = values.iter()
//!     .map(|v| encode_sortable(v).unwrap())
//!     .collect();
//!
//! // Sorting encoded bytes gives the same order as sorting values
//! encoded.sort();
//!
//! let decoded: Vec<_> = encoded.iter()
//!     .map(|e| decode_sortable(e).unwrap())
//!     .collect();
//!
//! assert_eq!(decoded, values);
//! ```

use crate::error::CoreError;
use crate::types::Value;

/// Type tags for sortable encoding.
///
/// These tags define the sort order of different types.
pub mod tags {
    /// Null values sort first.
    pub const NULL: u8 = 0x00;
    /// Boolean values (false=0x00, true=0x01).
    pub const BOOL: u8 = 0x01;
    /// 64-bit signed integers.
    pub const INT: u8 = 0x02;
    /// 64-bit floating point numbers.
    pub const FLOAT: u8 = 0x03;
    /// UTF-8 strings.
    pub const STRING: u8 = 0x04;
    /// Raw bytes.
    pub const BYTES: u8 = 0x05;
}

/// Constant for flipping the sign bit of signed integers.
const SIGN_FLIP_I64: u64 = 0x8000_0000_0000_0000;

/// Escape byte: when we see 0x00 in data, we output 0x00 0x01
const ESCAPE_BYTE: u8 = 0x01;
/// Terminator: end of string/bytes is marked by 0x00 0x00
const TERMINATOR: u8 = 0x00;

/// Encode bytes with null-escape encoding.
///
/// This encoding preserves lexicographic order:
/// - Each 0x00 in input becomes 0x00 0x01
/// - Sequence ends with 0x00 0x00
fn encode_bytes_escaped(data: &[u8], buf: &mut Vec<u8>) {
    for &byte in data {
        if byte == 0x00 {
            buf.push(0x00);
            buf.push(ESCAPE_BYTE);
        } else {
            buf.push(byte);
        }
    }
    // Terminator: 0x00 0x00
    buf.push(TERMINATOR);
    buf.push(TERMINATOR);
}

/// Decode bytes with null-escape encoding.
///
/// Returns the decoded bytes and the number of input bytes consumed.
fn decode_bytes_escaped(data: &[u8]) -> Result<(Vec<u8>, usize), CoreError> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0x00 {
            if i + 1 >= data.len() {
                return Err(CoreError::Encoding("unexpected end of escaped bytes".into()));
            }
            match data[i + 1] {
                TERMINATOR => {
                    // End of sequence
                    return Ok((result, i + 2));
                }
                ESCAPE_BYTE => {
                    // Escaped null byte
                    result.push(0x00);
                    i += 2;
                }
                other => {
                    return Err(CoreError::Encoding(format!(
                        "invalid escape sequence: 0x00 0x{other:02x}"
                    )));
                }
            }
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    Err(CoreError::Encoding("missing terminator in escaped bytes".into()))
}

/// Encode a value into a sort-order preserving byte representation.
///
/// The encoded bytes can be compared lexicographically to determine the
/// ordering of the original values.
///
/// # Arguments
///
/// * `value` - The value to encode
///
/// # Returns
///
/// A `Vec<u8>` containing the sortable encoding.
///
/// # Errors
///
/// Returns [`CoreError::Encoding`] if the value type is not supported for
/// indexing (e.g., vectors, arrays).
///
/// # Example
///
/// ```
/// use manifoldb_core::encoding::sortable::encode_sortable;
/// use manifoldb_core::types::Value;
///
/// let encoded_neg = encode_sortable(&Value::Int(-5)).unwrap();
/// let encoded_pos = encode_sortable(&Value::Int(5)).unwrap();
///
/// // Negative numbers sort before positive numbers
/// assert!(encoded_neg < encoded_pos);
/// ```
pub fn encode_sortable(value: &Value) -> Result<Vec<u8>, CoreError> {
    match value {
        Value::Null => Ok(vec![tags::NULL]),

        Value::Bool(b) => Ok(vec![tags::BOOL, u8::from(*b)]),

        Value::Int(i) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(tags::INT);
            // Flip sign bit to make negative numbers sort before positive
            let encoded = (*i as u64) ^ SIGN_FLIP_I64;
            buf.extend_from_slice(&encoded.to_be_bytes());
            Ok(buf)
        }

        Value::Float(f) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(tags::FLOAT);
            let bits = f.to_bits();
            // Handle NaN: encode as maximum value so it sorts last
            let encoded = if f.is_nan() {
                u64::MAX
            } else if bits & SIGN_FLIP_I64 == 0 {
                // Positive float (including +0): flip sign bit
                bits ^ SIGN_FLIP_I64
            } else {
                // Negative float (including -0): flip all bits
                !bits
            };
            buf.extend_from_slice(&encoded.to_be_bytes());
            Ok(buf)
        }

        Value::String(s) => {
            let bytes = s.as_bytes();
            // Estimate capacity: tag + data + worst case escaping + terminator
            let mut buf = Vec::with_capacity(1 + bytes.len() * 2 + 2);
            buf.push(tags::STRING);
            encode_bytes_escaped(bytes, &mut buf);
            Ok(buf)
        }

        Value::Bytes(b) => {
            // Estimate capacity: tag + data + worst case escaping + terminator
            let mut buf = Vec::with_capacity(1 + b.len() * 2 + 2);
            buf.push(tags::BYTES);
            encode_bytes_escaped(b, &mut buf);
            Ok(buf)
        }

        Value::Vector(_) | Value::SparseVector(_) | Value::MultiVector(_) | Value::Array(_) => {
            Err(CoreError::Encoding(
                "vectors and arrays are not supported for sortable encoding".into(),
            ))
        }
    }
}

/// Decode a sortable-encoded value back to its original form.
///
/// # Arguments
///
/// * `bytes` - The encoded bytes to decode
///
/// # Returns
///
/// The decoded [`Value`].
///
/// # Errors
///
/// Returns [`CoreError::Encoding`] if the bytes are malformed or incomplete.
///
/// # Example
///
/// ```
/// use manifoldb_core::encoding::sortable::{encode_sortable, decode_sortable};
/// use manifoldb_core::types::Value;
///
/// let original = Value::Int(42);
/// let encoded = encode_sortable(&original).unwrap();
/// let decoded = decode_sortable(&encoded).unwrap();
///
/// assert_eq!(decoded, original);
/// ```
pub fn decode_sortable(bytes: &[u8]) -> Result<Value, CoreError> {
    let (value, _) = decode_sortable_with_len(bytes)?;
    Ok(value)
}

/// Decode a sortable-encoded value and return the number of bytes consumed.
///
/// This is useful when the encoded value is part of a larger key.
///
/// # Arguments
///
/// * `bytes` - The encoded bytes to decode
///
/// # Returns
///
/// A tuple of the decoded [`Value`] and the number of bytes consumed.
///
/// # Errors
///
/// Returns [`CoreError::Encoding`] if the bytes are malformed or incomplete.
pub fn decode_sortable_with_len(bytes: &[u8]) -> Result<(Value, usize), CoreError> {
    if bytes.is_empty() {
        return Err(CoreError::Encoding("unexpected end of input in sortable decode".into()));
    }

    let tag = bytes[0];
    let rest = &bytes[1..];

    match tag {
        tags::NULL => Ok((Value::Null, 1)),

        tags::BOOL => {
            if rest.is_empty() {
                return Err(CoreError::Encoding("unexpected end of input reading bool".into()));
            }
            Ok((Value::Bool(rest[0] != 0), 2))
        }

        tags::INT => {
            if rest.len() < 8 {
                return Err(CoreError::Encoding("unexpected end of input reading int".into()));
            }
            let encoded_bytes: [u8; 8] = rest[..8]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read int bytes".into()))?;
            let encoded = u64::from_be_bytes(encoded_bytes);
            // Flip sign bit back
            let value = (encoded ^ SIGN_FLIP_I64) as i64;
            Ok((Value::Int(value), 9))
        }

        tags::FLOAT => {
            if rest.len() < 8 {
                return Err(CoreError::Encoding("unexpected end of input reading float".into()));
            }
            let encoded_bytes: [u8; 8] = rest[..8]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read float bytes".into()))?;
            let encoded = u64::from_be_bytes(encoded_bytes);

            // Decode based on the encoding rules
            let bits = if encoded == u64::MAX {
                // NaN was encoded as MAX
                f64::NAN.to_bits()
            } else if encoded & SIGN_FLIP_I64 != 0 {
                // Was positive (sign bit is set after XOR): flip sign bit back
                encoded ^ SIGN_FLIP_I64
            } else {
                // Was negative (sign bit is clear): flip all bits back
                !encoded
            };
            Ok((Value::Float(f64::from_bits(bits)), 9))
        }

        tags::STRING => {
            let (decoded_bytes, consumed) = decode_bytes_escaped(rest)?;
            let s = String::from_utf8(decoded_bytes)
                .map_err(|e| CoreError::Encoding(format!("invalid UTF-8: {e}")))?;
            Ok((Value::String(s), 1 + consumed))
        }

        tags::BYTES => {
            let (decoded_bytes, consumed) = decode_bytes_escaped(rest)?;
            Ok((Value::Bytes(decoded_bytes), 1 + consumed))
        }

        _ => Err(CoreError::Encoding(format!("unknown sortable type tag: {tag:#x}"))),
    }
}

/// Compute the estimated size of a sortable-encoded value.
///
/// This is useful for pre-allocating buffers. Note that for strings and bytes,
/// the actual size may be larger if the data contains null bytes (which get escaped).
///
/// # Arguments
///
/// * `value` - The value to compute the size for
///
/// # Returns
///
/// The estimated minimum size in bytes, or `None` if the value type is not supported.
#[must_use]
pub fn sortable_encoded_size(value: &Value) -> Option<usize> {
    match value {
        Value::Null => Some(1),
        Value::Bool(_) => Some(2),
        Value::Int(_) => Some(9),
        Value::Float(_) => Some(9),
        // tag(1) + data + terminator(2), data may expand due to escaping
        Value::String(s) => Some(1 + s.len() + 2),
        Value::Bytes(b) => Some(1 + b.len() + 2),
        Value::Vector(_) | Value::SparseVector(_) | Value::MultiVector(_) | Value::Array(_) => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ========================================================================
    // Round-trip tests
    // ========================================================================

    #[test]
    fn roundtrip_null() {
        let original = Value::Null;
        let encoded = encode_sortable(&original).unwrap();
        let decoded = decode_sortable(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn roundtrip_bool() {
        for b in [false, true] {
            let original = Value::Bool(b);
            let encoded = encode_sortable(&original).unwrap();
            let decoded = decode_sortable(&encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn roundtrip_int() {
        for i in [i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX] {
            let original = Value::Int(i);
            let encoded = encode_sortable(&original).unwrap();
            let decoded = decode_sortable(&encoded).unwrap();
            assert_eq!(decoded, original, "failed for {i}");
        }
    }

    #[test]
    fn roundtrip_float() {
        for f in [f64::NEG_INFINITY, -1000.0, -1.0, -0.0, 0.0, 1.0, 1000.0, f64::INFINITY] {
            let original = Value::Float(f);
            let encoded = encode_sortable(&original).unwrap();
            let decoded = decode_sortable(&encoded).unwrap();
            assert_eq!(decoded, original, "failed for {f}");
        }
    }

    #[test]
    fn roundtrip_float_nan() {
        let original = Value::Float(f64::NAN);
        let encoded = encode_sortable(&original).unwrap();
        let decoded = decode_sortable(&encoded).unwrap();
        match decoded {
            Value::Float(f) => assert!(f.is_nan()),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn roundtrip_string() {
        for s in ["", "a", "hello", "hello world", "日本語", "\u{1F600}"] {
            let original = Value::String(s.to_owned());
            let encoded = encode_sortable(&original).unwrap();
            let decoded = decode_sortable(&encoded).unwrap();
            assert_eq!(decoded, original, "failed for {s:?}");
        }
    }

    #[test]
    fn roundtrip_bytes() {
        for b in [vec![], vec![0u8], vec![1, 2, 3], vec![255, 0, 128]] {
            let original = Value::Bytes(b.clone());
            let encoded = encode_sortable(&original).unwrap();
            let decoded = decode_sortable(&encoded).unwrap();
            assert_eq!(decoded, original, "failed for {b:?}");
        }
    }

    // ========================================================================
    // Sort order tests
    // ========================================================================

    #[test]
    fn sort_order_types() {
        // Different types should sort by their type tag
        let values = vec![
            Value::Null,
            Value::Bool(false),
            Value::Int(0),
            Value::Float(0.0),
            Value::String(String::new()),
            Value::Bytes(vec![]),
        ];

        let mut encoded: Vec<_> = values.iter().map(|v| encode_sortable(v).unwrap()).collect();
        let original_order = encoded.clone();
        encoded.sort();

        assert_eq!(encoded, original_order, "type ordering should match");
    }

    #[test]
    fn sort_order_bool() {
        let false_enc = encode_sortable(&Value::Bool(false)).unwrap();
        let true_enc = encode_sortable(&Value::Bool(true)).unwrap();

        assert!(false_enc < true_enc, "false should sort before true");
    }

    #[test]
    fn sort_order_int() {
        let values: Vec<i64> = vec![i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX];

        let mut encoded: Vec<_> =
            values.iter().map(|i| encode_sortable(&Value::Int(*i)).unwrap()).collect();
        let original_order = encoded.clone();
        encoded.sort();

        assert_eq!(encoded, original_order, "integers should maintain sort order");
    }

    #[test]
    fn sort_order_float() {
        let values: Vec<f64> = vec![
            f64::NEG_INFINITY,
            -1000.0,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            1000.0,
            f64::INFINITY,
            f64::NAN, // NaN sorts last
        ];

        let mut encoded: Vec<_> =
            values.iter().map(|f| encode_sortable(&Value::Float(*f)).unwrap()).collect();
        let original_order = encoded.clone();
        encoded.sort();

        assert_eq!(encoded, original_order, "floats should maintain sort order");
    }

    #[test]
    fn sort_order_string() {
        let values: Vec<&str> = vec!["", "a", "aa", "ab", "b", "hello", "world"];

        let mut encoded: Vec<_> = values
            .iter()
            .map(|s| encode_sortable(&Value::String((*s).to_owned())).unwrap())
            .collect();
        let original_order = encoded.clone();
        encoded.sort();

        assert_eq!(encoded, original_order, "strings should maintain sort order");
    }

    #[test]
    fn sort_order_bytes() {
        let values: Vec<Vec<u8>> =
            vec![vec![], vec![0], vec![0, 0], vec![0, 1], vec![1], vec![1, 0], vec![255]];

        let mut encoded: Vec<_> =
            values.iter().map(|b| encode_sortable(&Value::Bytes(b.clone())).unwrap()).collect();
        let original_order = encoded.clone();
        encoded.sort();

        assert_eq!(encoded, original_order, "bytes should maintain sort order");
    }

    // ========================================================================
    // Error handling tests
    // ========================================================================

    #[test]
    fn encode_vector_fails() {
        let value = Value::Vector(vec![1.0, 2.0, 3.0]);
        assert!(encode_sortable(&value).is_err());
    }

    #[test]
    fn encode_sparse_vector_fails() {
        let value = Value::SparseVector(vec![(0, 1.0)]);
        assert!(encode_sortable(&value).is_err());
    }

    #[test]
    fn encode_multi_vector_fails() {
        let value = Value::MultiVector(vec![vec![1.0]]);
        assert!(encode_sortable(&value).is_err());
    }

    #[test]
    fn encode_array_fails() {
        let value = Value::Array(vec![Value::Int(1)]);
        assert!(encode_sortable(&value).is_err());
    }

    #[test]
    fn decode_empty_fails() {
        assert!(decode_sortable(&[]).is_err());
    }

    #[test]
    fn decode_truncated_int_fails() {
        let bytes = [tags::INT, 0, 0, 0]; // Only 4 bytes instead of 8
        assert!(decode_sortable(&bytes).is_err());
    }

    #[test]
    fn decode_truncated_string_fails() {
        // String without terminator
        let bytes = [tags::STRING, b'h', b'e', b'l', b'l', b'o'];
        assert!(decode_sortable(&bytes).is_err());
    }

    #[test]
    fn decode_unknown_tag_fails() {
        let bytes = [0xFF];
        assert!(decode_sortable(&bytes).is_err());
    }

    // ========================================================================
    // Size estimation tests
    // ========================================================================

    #[test]
    fn size_estimation() {
        assert_eq!(sortable_encoded_size(&Value::Null), Some(1));
        assert_eq!(sortable_encoded_size(&Value::Bool(true)), Some(2));
        assert_eq!(sortable_encoded_size(&Value::Int(42)), Some(9));
        assert_eq!(sortable_encoded_size(&Value::Float(3.14)), Some(9));
        // String: tag(1) + data(5) + terminator(2) = 8
        assert_eq!(sortable_encoded_size(&Value::String("hello".into())), Some(8));
        // Bytes: tag(1) + data(3) + terminator(2) = 6
        assert_eq!(sortable_encoded_size(&Value::Bytes(vec![1, 2, 3])), Some(6));
        assert_eq!(sortable_encoded_size(&Value::Vector(vec![1.0])), None);
        assert_eq!(sortable_encoded_size(&Value::Array(vec![])), None);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn int_boundary_values() {
        // Test values around 0 and boundaries
        let test_values = vec![i64::MIN, i64::MIN + 1, -2, -1, 0, 1, 2, i64::MAX - 1, i64::MAX];

        for i in 0..test_values.len() - 1 {
            let enc_a = encode_sortable(&Value::Int(test_values[i])).unwrap();
            let enc_b = encode_sortable(&Value::Int(test_values[i + 1])).unwrap();
            assert!(enc_a < enc_b, "{} should sort before {}", test_values[i], test_values[i + 1]);
        }
    }

    #[test]
    fn float_special_values() {
        // Test -0.0 vs +0.0
        let neg_zero = encode_sortable(&Value::Float(-0.0)).unwrap();
        let pos_zero = encode_sortable(&Value::Float(0.0)).unwrap();
        // -0.0 should sort before +0.0
        assert!(neg_zero < pos_zero);

        // Test subnormal numbers
        let subnormal = 5e-324_f64;
        let normal = 1e-300_f64;
        let enc_sub = encode_sortable(&Value::Float(subnormal)).unwrap();
        let enc_norm = encode_sortable(&Value::Float(normal)).unwrap();
        assert!(enc_sub < enc_norm);
    }

    #[test]
    fn decode_with_trailing_bytes() {
        // decode_sortable_with_len should correctly report consumed bytes
        let value = Value::Int(42);
        let mut encoded = encode_sortable(&value).unwrap();
        encoded.extend_from_slice(b"trailing");

        let (decoded, consumed) = decode_sortable_with_len(&encoded).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(consumed, 9);
        assert_eq!(&encoded[consumed..], b"trailing");
    }
}
