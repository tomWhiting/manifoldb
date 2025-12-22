//! Sparse vector type for efficient storage and operations.
//!
//! Sparse vectors are represented as sorted (index, value) pairs, storing only
//! non-zero values. This is efficient for high-dimensional vectors with few
//! non-zero elements, such as SPLADE embeddings or sparse retrievers.

use std::ops::Deref;

use crate::error::VectorError;

/// A sparse vector embedding with validation.
///
/// Sparse vectors store only non-zero values as (index, value) pairs.
/// The pairs are always sorted by index in ascending order.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::SparseEmbedding;
///
/// // Create from unsorted pairs - they will be sorted
/// let embedding = SparseEmbedding::new(vec![(100, 0.5), (10, 0.3), (50, 0.2)]).unwrap();
/// assert_eq!(embedding.nnz(), 3);
///
/// // Indices are now sorted
/// assert_eq!(embedding.indices(), &[10, 50, 100]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SparseEmbedding {
    /// Sorted (index, value) pairs.
    data: Vec<(u32, f32)>,
    /// Maximum dimension (exclusive upper bound on indices).
    dimension: Option<u32>,
}

impl SparseEmbedding {
    /// Create a new sparse embedding from (index, value) pairs.
    ///
    /// The pairs will be sorted by index. Duplicate indices will cause an error.
    /// Zero values will be filtered out.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any value is NaN or Infinite
    /// - There are duplicate indices
    pub fn new(mut data: Vec<(u32, f32)>) -> Result<Self, VectorError> {
        // Validate values
        for (i, &(idx, value)) in data.iter().enumerate() {
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
            let _ = idx; // Silence unused warning
        }

        // Filter out zero values
        data.retain(|&(_, v)| v != 0.0);

        // Sort by index
        data.sort_by_key(|&(idx, _)| idx);

        // Check for duplicate indices
        for window in data.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(VectorError::Encoding(format!(
                    "duplicate index {} in sparse vector",
                    window[0].0
                )));
            }
        }

        Ok(Self { data, dimension: None })
    }

    /// Create a sparse embedding with a specified maximum dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if any index is >= dimension, or the same errors as `new`.
    pub fn with_dimension(data: Vec<(u32, f32)>, dimension: u32) -> Result<Self, VectorError> {
        // Check indices are within bounds
        for &(idx, _) in &data {
            if idx >= dimension {
                return Err(VectorError::Encoding(format!(
                    "index {} exceeds dimension {}",
                    idx, dimension
                )));
            }
        }

        let mut embedding = Self::new(data)?;
        embedding.dimension = Some(dimension);
        Ok(embedding)
    }

    /// Create from raw bytes (compressed format).
    ///
    /// Format: 4 bytes (count) + for each entry: 4 bytes (u32 index) + 4 bytes (f32 value)
    ///
    /// # Errors
    ///
    /// Returns an error if the byte format is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.len() < 4 {
            return Err(VectorError::Encoding("sparse vector bytes too short".to_string()));
        }

        let count_bytes: [u8; 4] = bytes[..4]
            .try_into()
            .map_err(|_| VectorError::Encoding("failed to read count".to_string()))?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let expected_len = 4 + count * 8;
        if bytes.len() < expected_len {
            return Err(VectorError::Encoding(format!(
                "expected {} bytes, got {}",
                expected_len,
                bytes.len()
            )));
        }

        let mut data = Vec::with_capacity(count);
        for i in 0..count {
            let offset = 4 + i * 8;
            let idx_bytes: [u8; 4] = bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| VectorError::Encoding("failed to read index".to_string()))?;
            let val_bytes: [u8; 4] = bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| VectorError::Encoding("failed to read value".to_string()))?;

            let idx = u32::from_le_bytes(idx_bytes);
            let val = f32::from_le_bytes(val_bytes);

            if !val.is_finite() {
                return Err(VectorError::InvalidValue {
                    index: i,
                    value: val,
                    reason: if val.is_nan() {
                        "NaN values are not allowed"
                    } else {
                        "Infinite values are not allowed"
                    },
                });
            }

            data.push((idx, val));
        }

        // Data should already be sorted, but verify
        for window in data.windows(2) {
            if window[0].0 >= window[1].0 {
                return Err(VectorError::Encoding("sparse vector indices not sorted".to_string()));
            }
        }

        Ok(Self { data, dimension: None })
    }

    /// Convert to raw bytes (compressed format).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.data.len() * 8);
        bytes.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        for &(idx, val) in &self.data {
            bytes.extend_from_slice(&idx.to_le_bytes());
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Create an empty sparse embedding.
    #[must_use]
    pub const fn empty() -> Self {
        Self { data: Vec::new(), dimension: None }
    }

    /// Get the number of non-zero elements.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty (all zeros).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the maximum dimension if specified.
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> Option<u32> {
        self.dimension
    }

    /// Get the indices of non-zero elements.
    #[must_use]
    pub fn indices(&self) -> Vec<u32> {
        self.data.iter().map(|&(idx, _)| idx).collect()
    }

    /// Get the values of non-zero elements.
    #[must_use]
    pub fn values(&self) -> Vec<f32> {
        self.data.iter().map(|&(_, val)| val).collect()
    }

    /// Get the data as a slice of (index, value) pairs.
    #[inline]
    #[must_use]
    pub fn as_pairs(&self) -> &[(u32, f32)] {
        &self.data
    }

    /// Get the value at a specific index.
    ///
    /// Returns 0.0 if the index is not present.
    #[must_use]
    pub fn get(&self, index: u32) -> f32 {
        self.data
            .binary_search_by_key(&index, |&(idx, _)| idx)
            .map(|i| self.data[i].1)
            .unwrap_or(0.0)
    }

    /// Consume the embedding and return the underlying data.
    #[inline]
    #[must_use]
    pub fn into_pairs(self) -> Vec<(u32, f32)> {
        self.data
    }

    /// Calculate the L2 (Euclidean) norm of the sparse vector.
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|&(_, v)| v * v).sum::<f32>().sqrt()
    }

    /// Calculate the sum of squares (squared L2 norm).
    #[must_use]
    pub fn sum_of_squares(&self) -> f32 {
        self.data.iter().map(|&(_, v)| v * v).sum()
    }

    /// Normalize the sparse vector to unit length (L2 norm = 1).
    ///
    /// Returns a new normalized embedding. If the embedding has zero length,
    /// returns a copy of the original.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let norm = self.l2_norm();
        if norm == 0.0 {
            return self.clone();
        }

        let data: Vec<(u32, f32)> = self.data.iter().map(|&(idx, v)| (idx, v / norm)).collect();

        Self { data, dimension: self.dimension }
    }

    /// Scale all values by a constant factor.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        let data: Vec<(u32, f32)> = self
            .data
            .iter()
            .map(|&(idx, v)| (idx, v * factor))
            .filter(|&(_, v)| v != 0.0)
            .collect();

        Self { data, dimension: self.dimension }
    }

    /// Add two sparse vectors element-wise.
    ///
    /// The result contains the union of indices from both vectors.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.data.len() + other.data.len());

        let mut i = 0;
        let mut j = 0;

        while i < self.data.len() && j < other.data.len() {
            let (idx_a, val_a) = self.data[i];
            let (idx_b, val_b) = other.data[j];

            match idx_a.cmp(&idx_b) {
                std::cmp::Ordering::Less => {
                    result.push((idx_a, val_a));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push((idx_b, val_b));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let sum = val_a + val_b;
                    if sum != 0.0 {
                        result.push((idx_a, sum));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        // Add remaining elements
        result.extend_from_slice(&self.data[i..]);
        result.extend_from_slice(&other.data[j..]);

        Self {
            data: result,
            dimension: match (self.dimension, other.dimension) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
        }
    }
}

impl Deref for SparseEmbedding {
    type Target = [(u32, f32)];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl AsRef<[(u32, f32)]> for SparseEmbedding {
    #[inline]
    fn as_ref(&self) -> &[(u32, f32)] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sparse_embedding() {
        let embedding = SparseEmbedding::new(vec![(10, 0.5), (5, 0.3), (20, 0.2)]).unwrap();
        assert_eq!(embedding.nnz(), 3);
        // Should be sorted
        assert_eq!(embedding.indices(), vec![5, 10, 20]);
        assert_eq!(embedding.values(), vec![0.3, 0.5, 0.2]);
    }

    #[test]
    fn empty_sparse_embedding() {
        let embedding = SparseEmbedding::empty();
        assert!(embedding.is_empty());
        assert_eq!(embedding.nnz(), 0);
    }

    #[test]
    fn sparse_embedding_filters_zeros() {
        let embedding = SparseEmbedding::new(vec![(0, 0.5), (1, 0.0), (2, 0.3)]).unwrap();
        assert_eq!(embedding.nnz(), 2);
        assert_eq!(embedding.indices(), vec![0, 2]);
    }

    #[test]
    fn sparse_embedding_rejects_nan() {
        let result = SparseEmbedding::new(vec![(0, f32::NAN)]);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_embedding_rejects_infinity() {
        let result = SparseEmbedding::new(vec![(0, f32::INFINITY)]);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_embedding_rejects_duplicates() {
        let result = SparseEmbedding::new(vec![(0, 0.5), (0, 0.3)]);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_embedding_with_dimension() {
        let embedding = SparseEmbedding::with_dimension(vec![(0, 0.5), (99, 0.3)], 100).unwrap();
        assert_eq!(embedding.dimension(), Some(100));
    }

    #[test]
    fn sparse_embedding_dimension_check() {
        let result = SparseEmbedding::with_dimension(vec![(100, 0.5)], 100);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_embedding_get() {
        let embedding = SparseEmbedding::new(vec![(5, 0.5), (10, 0.3)]).unwrap();
        assert!((embedding.get(5) - 0.5).abs() < 1e-6);
        assert!((embedding.get(10) - 0.3).abs() < 1e-6);
        assert!((embedding.get(0) - 0.0).abs() < 1e-6);
        assert!((embedding.get(7) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_embedding_l2_norm() {
        let embedding = SparseEmbedding::new(vec![(0, 3.0), (1, 4.0)]).unwrap();
        assert!((embedding.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_embedding_normalize() {
        let embedding = SparseEmbedding::new(vec![(0, 3.0), (1, 4.0)]).unwrap();
        let normalized = embedding.normalize();
        assert!((normalized.l2_norm() - 1.0).abs() < 1e-6);
        assert!((normalized.get(0) - 0.6).abs() < 1e-6);
        assert!((normalized.get(1) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sparse_embedding_normalize_zero() {
        let embedding = SparseEmbedding::empty();
        let normalized = embedding.normalize();
        assert!(normalized.is_empty());
    }

    #[test]
    fn sparse_embedding_scale() {
        let embedding = SparseEmbedding::new(vec![(0, 1.0), (1, 2.0)]).unwrap();
        let scaled = embedding.scale(2.0);
        assert!((scaled.get(0) - 2.0).abs() < 1e-6);
        assert!((scaled.get(1) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_embedding_add() {
        let a = SparseEmbedding::new(vec![(0, 1.0), (2, 2.0)]).unwrap();
        let b = SparseEmbedding::new(vec![(1, 1.0), (2, 3.0)]).unwrap();
        let sum = a.add(&b);
        assert_eq!(sum.nnz(), 3);
        assert!((sum.get(0) - 1.0).abs() < 1e-6);
        assert!((sum.get(1) - 1.0).abs() < 1e-6);
        assert!((sum.get(2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_embedding_bytes_roundtrip() {
        let original = SparseEmbedding::new(vec![(0, 0.5), (10, 0.3), (100, 0.2)]).unwrap();
        let bytes = original.to_bytes();
        let restored = SparseEmbedding::from_bytes(&bytes).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn sparse_embedding_empty_bytes() {
        let original = SparseEmbedding::empty();
        let bytes = original.to_bytes();
        let restored = SparseEmbedding::from_bytes(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn sparse_embedding_deref() {
        let embedding = SparseEmbedding::new(vec![(5, 0.5), (10, 0.3)]).unwrap();
        let slice: &[(u32, f32)] = &embedding;
        assert_eq!(slice.len(), 2);
    }
}
