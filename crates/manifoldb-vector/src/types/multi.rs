//! Multi-vector embedding type for ColBERT-style late interaction models.
//!
//! Multi-vectors store per-token embeddings as a sequence of dense vectors.
//! This is used for late interaction models like ColBERT, where similarity
//! is computed using MaxSim over token pairs.

use std::ops::Deref;

use crate::error::VectorError;

/// A multi-vector embedding with validation.
///
/// Multi-vector embeddings store a sequence of dense vectors, where each vector
/// represents a token embedding. This is used for ColBERT-style late interaction
/// models where similarity is computed using MaxSim:
///
/// ```text
/// MaxSim(Q, D) = sum over query tokens q_i of max over doc tokens d_j of (q_i · d_j)
/// ```
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::MultiVectorEmbedding;
///
/// // Create from token embeddings
/// let embeddings = vec![
///     vec![0.1, 0.2, 0.3],  // Token 1 embedding
///     vec![0.4, 0.5, 0.6],  // Token 2 embedding
///     vec![0.7, 0.8, 0.9],  // Token 3 embedding
/// ];
/// let multi = MultiVectorEmbedding::new(embeddings).unwrap();
/// assert_eq!(multi.num_vectors(), 3);
/// assert_eq!(multi.dimension(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultiVectorEmbedding {
    /// The token embeddings, stored as a flat array for SIMD-friendly layout.
    data: Vec<f32>,
    /// The dimension of each token embedding.
    dimension: usize,
    /// The number of token embeddings.
    num_vectors: usize,
}

impl MultiVectorEmbedding {
    /// Create a new multi-vector embedding from a list of token embeddings.
    ///
    /// All token embeddings must have the same dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embeddings list is empty
    /// - Token embeddings have inconsistent dimensions
    /// - Any value is NaN or Infinite
    pub fn new(embeddings: Vec<Vec<f32>>) -> Result<Self, VectorError> {
        if embeddings.is_empty() {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        let dimension = embeddings[0].len();
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        // Validate all embeddings have the same dimension
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: dimension,
                    actual: emb.len(),
                });
            }

            // Validate values
            for (j, &value) in emb.iter().enumerate() {
                if !value.is_finite() {
                    return Err(VectorError::InvalidValue {
                        index: i * dimension + j,
                        value,
                        reason: if value.is_nan() {
                            "NaN values are not allowed"
                        } else {
                            "Infinite values are not allowed"
                        },
                    });
                }
            }
        }

        let num_vectors = embeddings.len();
        let data: Vec<f32> = embeddings.into_iter().flatten().collect();

        Ok(Self { data, dimension, num_vectors })
    }

    /// Create from a flat array of f32 values with known dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the data length is not divisible by dimension,
    /// or if dimension is 0.
    pub fn from_flat(data: Vec<f32>, dimension: usize) -> Result<Self, VectorError> {
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        if data.len() % dimension != 0 {
            return Err(VectorError::Encoding(format!(
                "data length {} is not divisible by dimension {}",
                data.len(),
                dimension
            )));
        }

        let num_vectors = data.len() / dimension;
        if num_vectors == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        // Validate values
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

        Ok(Self { data, dimension, num_vectors })
    }

    /// Create from raw bytes (little-endian f32 values).
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length is not a multiple of 4 or is empty.
    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Result<Self, VectorError> {
        if bytes.is_empty() {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        if bytes.len() % 4 != 0 {
            return Err(VectorError::Encoding(format!(
                "byte length {} is not a multiple of 4",
                bytes.len()
            )));
        }

        let total_floats = bytes.len() / 4;
        if total_floats % dimension != 0 {
            return Err(VectorError::Encoding(format!(
                "float count {} is not divisible by dimension {}",
                total_floats, dimension
            )));
        }

        let mut data = Vec::with_capacity(total_floats);
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

        Self::from_flat(data, dimension)
    }

    /// Get the dimension of each token embedding.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of token embeddings.
    #[inline]
    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Get the total number of f32 values.
    #[inline]
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    /// Get the underlying flat data.
    #[inline]
    #[must_use]
    pub fn as_flat_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get a specific token embedding as a slice.
    #[inline]
    #[must_use]
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        if index >= self.num_vectors {
            return None;
        }
        let start = index * self.dimension;
        let end = start + self.dimension;
        Some(&self.data[start..end])
    }

    /// Iterate over token embeddings.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.dimension)
    }

    /// Convert to raw bytes (little-endian f32 values).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &value in &self.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Consume the embedding and return the underlying flat data.
    #[inline]
    #[must_use]
    pub fn into_flat_vec(self) -> Vec<f32> {
        self.data
    }

    /// Consume the embedding and return as a list of vectors.
    #[must_use]
    pub fn into_vecs(self) -> Vec<Vec<f32>> {
        self.data.chunks_exact(self.dimension).map(|chunk| chunk.to_vec()).collect()
    }

    /// Normalize all token embeddings to unit length (L2 norm = 1).
    ///
    /// Returns a new normalized multi-vector embedding.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());

        for chunk in self.data.chunks_exact(self.dimension) {
            let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                data.extend_from_slice(chunk);
            } else {
                data.extend(chunk.iter().map(|x| x / norm));
            }
        }

        Self { data, dimension: self.dimension, num_vectors: self.num_vectors }
    }
}

impl Deref for MultiVectorEmbedding {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl AsRef<[f32]> for MultiVectorEmbedding {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_multi_vector() {
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let multi = MultiVectorEmbedding::new(embeddings).unwrap();

        assert_eq!(multi.dimension(), 3);
        assert_eq!(multi.num_vectors(), 2);
        assert_eq!(multi.total_elements(), 6);
    }

    #[test]
    fn get_vector() {
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let multi = MultiVectorEmbedding::new(embeddings).unwrap();

        assert_eq!(multi.get_vector(0), Some([0.1, 0.2, 0.3].as_slice()));
        assert_eq!(multi.get_vector(1), Some([0.4, 0.5, 0.6].as_slice()));
        assert_eq!(multi.get_vector(2), None);
    }

    #[test]
    fn iterate_vectors() {
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let multi = MultiVectorEmbedding::new(embeddings.clone()).unwrap();

        let collected: Vec<_> = multi.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], [0.1, 0.2].as_slice());
        assert_eq!(collected[1], [0.3, 0.4].as_slice());
        assert_eq!(collected[2], [0.5, 0.6].as_slice());
    }

    #[test]
    fn empty_embeddings_fails() {
        let result = MultiVectorEmbedding::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn empty_vector_fails() {
        let result = MultiVectorEmbedding::new(vec![vec![]]);
        assert!(result.is_err());
    }

    #[test]
    fn dimension_mismatch_fails() {
        let result = MultiVectorEmbedding::new(vec![vec![0.1, 0.2], vec![0.3, 0.4, 0.5]]);
        assert!(result.is_err());
    }

    #[test]
    fn nan_fails() {
        let result = MultiVectorEmbedding::new(vec![vec![0.1, f32::NAN]]);
        assert!(result.is_err());
    }

    #[test]
    fn infinity_fails() {
        let result = MultiVectorEmbedding::new(vec![vec![f32::INFINITY, 0.2]]);
        assert!(result.is_err());
    }

    #[test]
    fn from_flat() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let multi = MultiVectorEmbedding::from_flat(data, 3).unwrap();

        assert_eq!(multi.dimension(), 3);
        assert_eq!(multi.num_vectors(), 2);
        assert_eq!(multi.get_vector(0), Some([0.1, 0.2, 0.3].as_slice()));
        assert_eq!(multi.get_vector(1), Some([0.4, 0.5, 0.6].as_slice()));
    }

    #[test]
    fn from_flat_invalid_size() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = MultiVectorEmbedding::from_flat(data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn bytes_roundtrip() {
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let original = MultiVectorEmbedding::new(embeddings).unwrap();
        let dimension = original.dimension();

        let bytes = original.to_bytes();
        let restored = MultiVectorEmbedding::from_bytes(&bytes, dimension).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn normalize() {
        let embeddings = vec![vec![3.0, 4.0], vec![0.0, 5.0]];
        let multi = MultiVectorEmbedding::new(embeddings).unwrap();
        let normalized = multi.normalize();

        // First vector: [3/5, 4/5] = [0.6, 0.8]
        let v0 = normalized.get_vector(0).unwrap();
        assert!((v0[0] - 0.6).abs() < 1e-6);
        assert!((v0[1] - 0.8).abs() < 1e-6);

        // Second vector: [0, 1]
        let v1 = normalized.get_vector(1).unwrap();
        assert!((v1[0] - 0.0).abs() < 1e-6);
        assert!((v1[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn into_vecs() {
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let multi = MultiVectorEmbedding::new(embeddings.clone()).unwrap();
        let vecs = multi.into_vecs();

        assert_eq!(vecs, embeddings);
    }
}
