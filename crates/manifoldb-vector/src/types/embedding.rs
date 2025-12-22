//! Embedding type for vector storage.

use std::ops::Deref;

use crate::error::VectorError;

/// A vector embedding with dimension validation.
///
/// Embeddings are fixed-dimension vectors of f32 values that can be used for
/// similarity search. The embedding is stored as a contiguous array of f32
/// values for SIMD-friendly memory layout.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::Embedding;
///
/// let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(embedding.dimension(), 3);
/// assert_eq!(embedding.as_slice(), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    data: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding from a vector of f32 values.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector is empty or contains NaN/Infinite values.
    pub fn new(data: Vec<f32>) -> Result<Self, VectorError> {
        if data.is_empty() {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        // Check for NaN or Infinite values
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(VectorError::InvalidValue {
                    index: i,
                    value,
                    reason: if value.is_nan() {
                        "NaN values are not allowed"
                    } else {
                        "Infinite values are not allowed"
                    },
                });
            }
        }

        Ok(Self { data })
    }

    /// Create an embedding from raw bytes (little-endian f32 values).
    ///
    /// This is used for deserialization from storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length is not a multiple of 4 or is empty.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.is_empty() {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        if bytes.len() % 4 != 0 {
            return Err(VectorError::Encoding(format!(
                "byte length {} is not a multiple of 4",
                bytes.len()
            )));
        }

        let dimension = bytes.len() / 4;
        let mut data = Vec::with_capacity(dimension);

        for chunk in bytes.chunks_exact(4) {
            let bytes_array: [u8; 4] = chunk
                .try_into()
                .map_err(|_| VectorError::Encoding("failed to read f32 bytes".to_string()))?;
            let value = f32::from_le_bytes(bytes_array);

            if !value.is_finite() {
                return Err(VectorError::InvalidValue {
                    index: data.len(),
                    value,
                    reason: if value.is_nan() {
                        "NaN values are not allowed"
                    } else {
                        "Infinite values are not allowed"
                    },
                });
            }

            data.push(value);
        }

        Ok(Self { data })
    }

    /// Create a zero-filled embedding of the given dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if dimension is 0.
    pub fn zeros(dimension: usize) -> Result<Self, VectorError> {
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }
        Ok(Self { data: vec![0.0; dimension] })
    }

    /// Get the dimension of the embedding.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Get the embedding data as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Convert the embedding to raw bytes (little-endian f32 values).
    ///
    /// This is used for serialization to storage.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &value in &self.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Consume the embedding and return the underlying vector.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Normalize the embedding to unit length (L2 norm = 1).
    ///
    /// Returns a new normalized embedding. If the embedding has zero length,
    /// returns a copy of the original.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return self.clone();
        }

        let data: Vec<f32> = self.data.iter().map(|x| x / norm).collect();
        Self { data }
    }

    /// Calculate the L2 (Euclidean) norm of the embedding.
    #[inline]
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl Deref for Embedding {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl AsRef<[f32]> for Embedding {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_embedding() {
        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(embedding.dimension(), 3);
        assert_eq!(embedding.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_embedding_empty_fails() {
        let result = Embedding::new(vec![]);
        assert!(result.is_err());
        match result.unwrap_err() {
            VectorError::InvalidDimension { expected, actual } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn new_embedding_nan_fails() {
        let result = Embedding::new(vec![1.0, f32::NAN, 3.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            VectorError::InvalidValue { index, reason, .. } => {
                assert_eq!(index, 1);
                assert!(reason.contains("NaN"));
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn new_embedding_infinity_fails() {
        let result = Embedding::new(vec![1.0, f32::INFINITY, 3.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            VectorError::InvalidValue { index, reason, .. } => {
                assert_eq!(index, 1);
                assert!(reason.contains("Infinite"));
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn zeros_embedding() {
        let embedding = Embedding::zeros(5).unwrap();
        assert_eq!(embedding.dimension(), 5);
        assert_eq!(embedding.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn zeros_dimension_zero_fails() {
        let result = Embedding::zeros(0);
        assert!(result.is_err());
    }

    #[test]
    fn bytes_roundtrip() {
        let original = Embedding::new(vec![1.0, -2.5, 3.5, 0.0]).unwrap();
        let bytes = original.to_bytes();
        let restored = Embedding::from_bytes(&bytes).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn from_bytes_empty_fails() {
        let result = Embedding::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_invalid_length_fails() {
        let result = Embedding::from_bytes(&[1, 2, 3]); // Not multiple of 4
        assert!(result.is_err());
    }

    #[test]
    fn normalize() {
        let embedding = Embedding::new(vec![3.0, 4.0]).unwrap();
        let normalized = embedding.normalize();

        // 3^2 + 4^2 = 25, sqrt(25) = 5
        // Normalized: [3/5, 4/5] = [0.6, 0.8]
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Check unit norm
        let norm = normalized.l2_norm();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector() {
        let embedding = Embedding::new(vec![0.0, 0.0, 0.0]).unwrap();
        let normalized = embedding.normalize();
        assert_eq!(normalized, embedding);
    }

    #[test]
    fn l2_norm() {
        let embedding = Embedding::new(vec![3.0, 4.0]).unwrap();
        assert!((embedding.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn into_vec() {
        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        let vec = embedding.into_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn deref_to_slice() {
        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        let slice: &[f32] = &embedding;
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }
}
