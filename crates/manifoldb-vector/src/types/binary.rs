//! Binary vector type for bit-packed embeddings.
//!
//! Binary vectors store embeddings as bit-packed `u64` values, enabling
//! extremely efficient storage and Hamming distance computation using
//! hardware popcount instructions.
//!
//! # Use Cases
//!
//! Binary embeddings are commonly produced by:
//! - Binary quantization (e.g., sign bit of float embeddings)
//! - Binary hashing (e.g., SimHash, MinHash)
//! - Specialized binary embedding models
//!
//! # Example
//!
//! ```
//! use manifoldb_vector::types::BinaryEmbedding;
//!
//! // Create from raw bits (128 bits = 2 u64s)
//! let bits = vec![0xFFFF_FFFF_0000_0000u64, 0x0000_FFFF_FFFF_0000u64];
//! let embedding = BinaryEmbedding::new(bits, 128).unwrap();
//!
//! assert_eq!(embedding.dimension(), 128);
//! assert_eq!(embedding.count_ones(), 64); // 32 + 32 ones
//! ```

use crate::error::VectorError;

/// A binary vector embedding represented as bit-packed `u64` values.
///
/// Each `u64` stores 64 bits, with the lowest bit index stored in the
/// least significant position. The dimension specifies the actual number
/// of bits used (which may not be a multiple of 64).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryEmbedding {
    /// Bit-packed data. Each u64 stores 64 bits.
    data: Vec<u64>,
    /// The actual number of bits in this embedding.
    dimension: usize,
}

impl BinaryEmbedding {
    /// Create a new binary embedding from bit-packed data.
    ///
    /// # Arguments
    ///
    /// * `data` - The bit-packed u64 values
    /// * `dimension` - The actual number of bits used
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dimension` is zero
    /// - `data.len()` is too small for the given dimension
    /// - `data.len()` is larger than necessary (contains unused u64s)
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_vector::types::BinaryEmbedding;
    ///
    /// // 64 bits fit in one u64
    /// let embedding = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64], 64).unwrap();
    ///
    /// // 65 bits need two u64s
    /// let embedding = BinaryEmbedding::new(vec![0u64, 1u64], 65).unwrap();
    /// ```
    pub fn new(data: Vec<u64>, dimension: usize) -> Result<Self, VectorError> {
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        let required_u64s = dimension.div_ceil(64);

        if data.len() < required_u64s {
            return Err(VectorError::DimensionMismatch {
                expected: required_u64s,
                actual: data.len(),
            });
        }

        if data.len() > required_u64s {
            return Err(VectorError::DimensionMismatch {
                expected: required_u64s,
                actual: data.len(),
            });
        }

        // Verify unused bits in the last u64 are zero
        let unused_bits = data.len() * 64 - dimension;
        if unused_bits > 0 {
            let mask = !0u64 << (64 - unused_bits);
            if data[data.len() - 1] & mask != 0 {
                return Err(VectorError::Encoding(
                    "unused bits in last u64 must be zero".to_string(),
                ));
            }
        }

        Ok(Self { data, dimension })
    }

    /// Create a binary embedding from a dense f32 vector using sign bits.
    ///
    /// Each positive (>= 0) value becomes 1, each negative value becomes 0.
    /// This is a common binary quantization technique.
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_vector::types::BinaryEmbedding;
    ///
    /// let floats = vec![0.5, -0.3, 0.0, 1.2];
    /// let binary = BinaryEmbedding::from_sign_bits(&floats).unwrap();
    ///
    /// assert_eq!(binary.dimension(), 4);
    /// assert!(binary.get_bit(0));  // 0.5 >= 0
    /// assert!(!binary.get_bit(1)); // -0.3 < 0
    /// assert!(binary.get_bit(2));  // 0.0 >= 0
    /// assert!(binary.get_bit(3));  // 1.2 >= 0
    /// ```
    pub fn from_sign_bits(floats: &[f32]) -> Result<Self, VectorError> {
        if floats.is_empty() {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        let dimension = floats.len();
        let num_u64s = dimension.div_ceil(64);
        let mut data = vec![0u64; num_u64s];

        for (i, &value) in floats.iter().enumerate() {
            if value >= 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                data[word_idx] |= 1u64 << bit_idx;
            }
        }

        Ok(Self { data, dimension })
    }

    /// Create a binary embedding with all zeros.
    ///
    /// # Errors
    ///
    /// Returns an error if `dimension` is zero.
    pub fn zeros(dimension: usize) -> Result<Self, VectorError> {
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        let num_u64s = dimension.div_ceil(64);
        Ok(Self { data: vec![0u64; num_u64s], dimension })
    }

    /// Create a binary embedding with all ones.
    ///
    /// # Errors
    ///
    /// Returns an error if `dimension` is zero.
    pub fn ones(dimension: usize) -> Result<Self, VectorError> {
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        let num_u64s = dimension.div_ceil(64);
        let mut data = vec![!0u64; num_u64s];

        // Clear unused bits in the last u64
        let used_bits_in_last = dimension % 64;
        if used_bits_in_last != 0 {
            data[num_u64s - 1] = (1u64 << used_bits_in_last) - 1;
        }

        Ok(Self { data, dimension })
    }

    /// Get the dimension (number of bits) of this embedding.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the raw bit-packed data.
    #[must_use]
    pub fn data(&self) -> &[u64] {
        &self.data
    }

    /// Get a specific bit by index.
    ///
    /// Returns `None` if `index >= dimension`.
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_vector::types::BinaryEmbedding;
    ///
    /// let embedding = BinaryEmbedding::new(vec![0b0000_0101u64], 8).unwrap();
    /// assert_eq!(embedding.try_get_bit(0), Some(true));
    /// assert_eq!(embedding.try_get_bit(1), Some(false));
    /// assert_eq!(embedding.try_get_bit(2), Some(true));
    /// assert_eq!(embedding.try_get_bit(100), None); // Out of bounds
    /// ```
    #[must_use]
    pub fn try_get_bit(&self, index: usize) -> Option<bool> {
        if index >= self.dimension {
            return None;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        Some((self.data[word_idx] >> bit_idx) & 1 == 1)
    }

    /// Get a specific bit by index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= dimension`. Use [`try_get_bit`](Self::try_get_bit)
    /// for a non-panicking alternative.
    #[must_use]
    pub fn get_bit(&self, index: usize) -> bool {
        self.try_get_bit(index).expect("bit index out of bounds")
    }

    /// Set a specific bit by index.
    ///
    /// Returns `Err` if `index >= dimension`.
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_vector::types::BinaryEmbedding;
    ///
    /// let mut embedding = BinaryEmbedding::zeros(8).unwrap();
    /// assert!(embedding.try_set_bit(0, true).is_ok());
    /// assert!(embedding.get_bit(0));
    /// assert!(embedding.try_set_bit(100, true).is_err()); // Out of bounds
    /// ```
    pub fn try_set_bit(&mut self, index: usize, value: bool) -> Result<(), VectorError> {
        if index >= self.dimension {
            return Err(VectorError::IndexOutOfBounds { index, max: self.dimension });
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[word_idx] &= !(1u64 << bit_idx);
        }
        Ok(())
    }

    /// Set a specific bit by index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= dimension`. Use [`try_set_bit`](Self::try_set_bit)
    /// for a non-panicking alternative.
    pub fn set_bit(&mut self, index: usize, value: bool) {
        self.try_set_bit(index, value).expect("bit index out of bounds");
    }

    /// Count the number of 1-bits in this embedding.
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        self.data.iter().map(|&w| w.count_ones()).sum()
    }

    /// Compute the XOR of this embedding with another.
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings have different dimensions.
    pub fn xor(&self, other: &Self) -> Result<Self, VectorError> {
        if self.dimension != other.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let data: Vec<u64> =
            self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a ^ b).collect();

        Ok(Self { data, dimension: self.dimension })
    }

    /// Compute the AND of this embedding with another.
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings have different dimensions.
    pub fn and(&self, other: &Self) -> Result<Self, VectorError> {
        if self.dimension != other.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let data: Vec<u64> =
            self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a & b).collect();

        Ok(Self { data, dimension: self.dimension })
    }

    /// Compute the OR of this embedding with another.
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings have different dimensions.
    pub fn or(&self, other: &Self) -> Result<Self, VectorError> {
        if self.dimension != other.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let data: Vec<u64> =
            self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a | b).collect();

        Ok(Self { data, dimension: self.dimension })
    }

    /// Encode the binary embedding to bytes.
    ///
    /// Format:
    /// - 1 byte: version
    /// - 4 bytes: dimension (big-endian u32)
    /// - N * 8 bytes: data (each u64 as big-endian)
    ///
    /// # Errors
    ///
    /// Returns an error if dimension exceeds u32::MAX.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VectorError> {
        let mut bytes = Vec::with_capacity(5 + self.data.len() * 8);

        // Version
        bytes.push(1);

        // Dimension
        let dim = u32::try_from(self.dimension).map_err(|_| {
            VectorError::Encoding(format!(
                "dimension too large: {} exceeds maximum of {}",
                self.dimension,
                u32::MAX
            ))
        })?;
        bytes.extend_from_slice(&dim.to_be_bytes());

        // Data
        for &word in &self.data {
            bytes.extend_from_slice(&word.to_be_bytes());
        }

        Ok(bytes)
    }

    /// Decode a binary embedding from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are invalid or truncated.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.is_empty() {
            return Err(VectorError::Encoding("empty binary embedding data".to_string()));
        }

        let version = bytes[0];
        if version != 1 {
            return Err(VectorError::Encoding(format!(
                "unsupported binary embedding version: {}",
                version
            )));
        }

        if bytes.len() < 5 {
            return Err(VectorError::Encoding("truncated binary embedding data".to_string()));
        }

        let dimension = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;

        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        let num_u64s = dimension.div_ceil(64);
        let expected_len = 5 + num_u64s * 8;

        if bytes.len() < expected_len {
            return Err(VectorError::Encoding(format!(
                "truncated binary embedding data: expected {} bytes, got {}",
                expected_len,
                bytes.len()
            )));
        }

        let mut data = Vec::with_capacity(num_u64s);
        for i in 0..num_u64s {
            let offset = 5 + i * 8;
            let word = u64::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            data.push(word);
        }

        Self::new(data, dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let embedding = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64], 64).unwrap();
        assert_eq!(embedding.dimension(), 64);
        assert_eq!(embedding.count_ones(), 64);
    }

    #[test]
    fn test_new_partial_word() {
        // 32 bits uses half of a u64, unused high bits must be zero
        let embedding = BinaryEmbedding::new(vec![0x0000_0000_FFFF_FFFFu64], 32).unwrap();
        assert_eq!(embedding.dimension(), 32);
        assert_eq!(embedding.count_ones(), 32);
    }

    #[test]
    fn test_new_unused_bits_not_zero_fails() {
        // High bits set but dimension is only 32
        let result = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64], 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_dimension_zero_fails() {
        let result = BinaryEmbedding::new(vec![], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_data_too_small_fails() {
        // 65 bits needs 2 u64s
        let result = BinaryEmbedding::new(vec![0u64], 65);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_sign_bits() {
        let floats = vec![0.5, -0.3, 0.0, -1.0, 1.0];
        let binary = BinaryEmbedding::from_sign_bits(&floats).unwrap();

        assert_eq!(binary.dimension(), 5);
        assert!(binary.get_bit(0)); // 0.5 >= 0
        assert!(!binary.get_bit(1)); // -0.3 < 0
        assert!(binary.get_bit(2)); // 0.0 >= 0
        assert!(!binary.get_bit(3)); // -1.0 < 0
        assert!(binary.get_bit(4)); // 1.0 >= 0
    }

    #[test]
    fn test_zeros() {
        let embedding = BinaryEmbedding::zeros(128).unwrap();
        assert_eq!(embedding.dimension(), 128);
        assert_eq!(embedding.count_ones(), 0);
    }

    #[test]
    fn test_ones() {
        let embedding = BinaryEmbedding::ones(128).unwrap();
        assert_eq!(embedding.dimension(), 128);
        assert_eq!(embedding.count_ones(), 128);
    }

    #[test]
    fn test_ones_partial() {
        let embedding = BinaryEmbedding::ones(100).unwrap();
        assert_eq!(embedding.dimension(), 100);
        assert_eq!(embedding.count_ones(), 100);
    }

    #[test]
    fn test_get_set_bit() {
        let mut embedding = BinaryEmbedding::zeros(128).unwrap();

        embedding.set_bit(0, true);
        embedding.set_bit(63, true);
        embedding.set_bit(64, true);
        embedding.set_bit(127, true);

        assert!(embedding.get_bit(0));
        assert!(embedding.get_bit(63));
        assert!(embedding.get_bit(64));
        assert!(embedding.get_bit(127));
        assert!(!embedding.get_bit(1));
        assert!(!embedding.get_bit(62));

        embedding.set_bit(0, false);
        assert!(!embedding.get_bit(0));
    }

    #[test]
    fn test_xor() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b1010_1010u64], 8).unwrap();
        let result = a.xor(&b).unwrap();

        // 1111_0000 ^ 1010_1010 = 0101_1010
        assert_eq!(result.data()[0], 0b0101_1010u64);
    }

    #[test]
    fn test_and() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b1010_1010u64], 8).unwrap();
        let result = a.and(&b).unwrap();

        // 1111_0000 & 1010_1010 = 1010_0000
        assert_eq!(result.data()[0], 0b1010_0000u64);
    }

    #[test]
    fn test_or() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b1010_1010u64], 8).unwrap();
        let result = a.or(&b).unwrap();

        // 1111_0000 | 1010_1010 = 1111_1010
        assert_eq!(result.data()[0], 0b1111_1010u64);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = BinaryEmbedding::zeros(64).unwrap();
        let b = BinaryEmbedding::zeros(128).unwrap();

        assert!(a.xor(&b).is_err());
        assert!(a.and(&b).is_err());
        assert!(a.or(&b).is_err());
    }

    #[test]
    fn test_roundtrip_bytes() {
        let original =
            BinaryEmbedding::new(vec![0xDEAD_BEEF_CAFE_BABEu64, 0x0123_4567_89AB_CDEFu64], 128)
                .unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = BinaryEmbedding::from_bytes(&bytes).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn test_roundtrip_bytes_partial() {
        let original = BinaryEmbedding::new(vec![0x0000_0000_FFFF_FFFFu64], 32).unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = BinaryEmbedding::from_bytes(&bytes).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn test_try_get_bit_valid() {
        let embedding = BinaryEmbedding::new(vec![0b0000_0101u64], 8).unwrap();
        assert_eq!(embedding.try_get_bit(0), Some(true));
        assert_eq!(embedding.try_get_bit(1), Some(false));
        assert_eq!(embedding.try_get_bit(2), Some(true));
        assert_eq!(embedding.try_get_bit(7), Some(false));
    }

    #[test]
    fn test_try_get_bit_out_of_bounds() {
        let embedding = BinaryEmbedding::new(vec![0b1111_1111u64], 8).unwrap();
        assert_eq!(embedding.try_get_bit(8), None);
        assert_eq!(embedding.try_get_bit(100), None);
        assert_eq!(embedding.try_get_bit(usize::MAX), None);
    }

    #[test]
    fn test_try_set_bit_valid() {
        let mut embedding = BinaryEmbedding::zeros(8).unwrap();

        assert!(embedding.try_set_bit(0, true).is_ok());
        assert!(embedding.get_bit(0));

        assert!(embedding.try_set_bit(7, true).is_ok());
        assert!(embedding.get_bit(7));

        assert!(embedding.try_set_bit(0, false).is_ok());
        assert!(!embedding.get_bit(0));
    }

    #[test]
    fn test_try_set_bit_out_of_bounds() {
        let mut embedding = BinaryEmbedding::zeros(8).unwrap();

        let result = embedding.try_set_bit(8, true);
        assert!(result.is_err());

        let result = embedding.try_set_bit(100, true);
        assert!(result.is_err());

        let result = embedding.try_set_bit(usize::MAX, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_single_bit() {
        let embedding = BinaryEmbedding::new(vec![1u64], 1).unwrap();
        assert_eq!(embedding.dimension(), 1);
        assert!(embedding.get_bit(0));
        assert_eq!(embedding.count_ones(), 1);
    }

    #[test]
    fn test_edge_case_max_single_word() {
        let embedding = BinaryEmbedding::ones(64).unwrap();
        assert_eq!(embedding.dimension(), 64);
        assert_eq!(embedding.count_ones(), 64);
        for i in 0..64 {
            assert!(embedding.get_bit(i));
        }
    }

    #[test]
    fn test_edge_case_boundary_bits() {
        // Test bits at word boundaries (63, 64, 127, 128)
        let mut embedding = BinaryEmbedding::zeros(256).unwrap();

        // Set bits at boundaries
        embedding.try_set_bit(63, true).unwrap();
        embedding.try_set_bit(64, true).unwrap();
        embedding.try_set_bit(127, true).unwrap();
        embedding.try_set_bit(128, true).unwrap();
        embedding.try_set_bit(255, true).unwrap();

        assert!(embedding.get_bit(63));
        assert!(embedding.get_bit(64));
        assert!(embedding.get_bit(127));
        assert!(embedding.get_bit(128));
        assert!(embedding.get_bit(255));

        // Check that neighboring bits are not affected
        assert!(!embedding.get_bit(62));
        assert!(!embedding.get_bit(65));
        assert!(!embedding.get_bit(126));
        assert!(!embedding.get_bit(129));
        assert!(!embedding.get_bit(254));
    }
}
