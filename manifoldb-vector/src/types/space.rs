//! Embedding space types.

use std::fmt;

use crate::distance::sparse::SparseDistanceMetric;
use crate::distance::DistanceMetric;
use crate::error::VectorError;

/// A name for an embedding space.
///
/// Embedding names identify different embedding spaces associated with entities.
/// For example, an entity might have `text_embedding` and `image_embedding` spaces.
///
/// Names must be non-empty and contain only alphanumeric characters, underscores,
/// and hyphens.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::EmbeddingName;
///
/// let name = EmbeddingName::new("text_embedding").unwrap();
/// assert_eq!(name.as_str(), "text_embedding");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingName(String);

impl EmbeddingName {
    /// Create a new embedding name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is empty or contains invalid characters.
    pub fn new(name: impl Into<String>) -> Result<Self, VectorError> {
        let name = name.into();

        if name.is_empty() {
            return Err(VectorError::InvalidName("embedding name cannot be empty".to_string()));
        }

        // Validate characters: alphanumeric, underscore, hyphen
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(VectorError::InvalidName(format!(
                "embedding name '{name}' contains invalid characters (allowed: alphanumeric, underscore, hyphen)"
            )));
        }

        Ok(Self(name))
    }

    /// Get the name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the name and return the underlying string.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for EmbeddingName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for EmbeddingName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Metadata about an embedding space.
///
/// An embedding space defines the properties of embeddings stored within it:
/// - The dimension of vectors in the space
/// - The distance metric used for similarity search
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::{EmbeddingName, EmbeddingSpace};
/// use manifoldb_vector::distance::DistanceMetric;
///
/// let name = EmbeddingName::new("text_embedding").unwrap();
/// let space = EmbeddingSpace::new(name, 384, DistanceMetric::Cosine);
///
/// assert_eq!(space.dimension(), 384);
/// assert_eq!(space.distance_metric(), DistanceMetric::Cosine);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingSpace {
    name: EmbeddingName,
    dimension: usize,
    distance_metric: DistanceMetric,
}

impl EmbeddingSpace {
    /// Create a new embedding space.
    #[must_use]
    pub const fn new(
        name: EmbeddingName,
        dimension: usize,
        distance_metric: DistanceMetric,
    ) -> Self {
        Self { name, dimension, distance_metric }
    }

    /// Get the name of the embedding space.
    #[must_use]
    pub fn name(&self) -> &EmbeddingName {
        &self.name
    }

    /// Get the dimension of embeddings in this space.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the distance metric used for similarity search.
    #[must_use]
    pub fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Encode the embedding space to bytes.
    ///
    /// Format:
    /// - 1 byte: version
    /// - 2 bytes: name length (big-endian u16)
    /// - N bytes: name (UTF-8)
    /// - 4 bytes: dimension (big-endian u32)
    /// - 1 byte: distance metric
    ///
    /// # Errors
    ///
    /// Returns an error if the name length exceeds u16::MAX or the dimension exceeds u32::MAX.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VectorError> {
        let name_bytes = self.name.as_str().as_bytes();
        let mut bytes = Vec::with_capacity(8 + name_bytes.len());

        // Version
        bytes.push(1);

        // Name length and name - fail if name is too long
        let name_len = u16::try_from(name_bytes.len()).map_err(|_| {
            VectorError::Encoding(format!(
                "embedding name too long: {} bytes exceeds maximum of {}",
                name_bytes.len(),
                u16::MAX
            ))
        })?;
        bytes.extend_from_slice(&name_len.to_be_bytes());
        bytes.extend_from_slice(name_bytes);

        // Dimension - fail if dimension is too large
        let dim = u32::try_from(self.dimension).map_err(|_| {
            VectorError::Encoding(format!(
                "embedding dimension too large: {} exceeds maximum of {}",
                self.dimension,
                u32::MAX
            ))
        })?;
        bytes.extend_from_slice(&dim.to_be_bytes());

        // Distance metric
        bytes.push(match self.distance_metric {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::DotProduct => 2,
        });

        Ok(bytes)
    }

    /// Decode an embedding space from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are invalid or truncated.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.is_empty() {
            return Err(VectorError::Encoding("empty embedding space data".to_string()));
        }

        let version = bytes[0];
        if version != 1 {
            return Err(VectorError::Encoding(format!(
                "unsupported embedding space version: {}",
                version
            )));
        }

        if bytes.len() < 3 {
            return Err(VectorError::Encoding("truncated embedding space data".to_string()));
        }

        // Name length
        let name_len = u16::from_be_bytes([bytes[1], bytes[2]]) as usize;

        if bytes.len() < 3 + name_len + 5 {
            return Err(VectorError::Encoding("truncated embedding space data".to_string()));
        }

        // Name
        let name_bytes = &bytes[3..3 + name_len];
        let name_str = std::str::from_utf8(name_bytes)
            .map_err(|e| VectorError::Encoding(format!("invalid UTF-8 in name: {}", e)))?;
        let name = EmbeddingName::new(name_str)?;

        // Dimension
        let dim_offset = 3 + name_len;
        let dimension = u32::from_be_bytes([
            bytes[dim_offset],
            bytes[dim_offset + 1],
            bytes[dim_offset + 2],
            bytes[dim_offset + 3],
        ]) as usize;

        // Distance metric
        let metric_byte = bytes[dim_offset + 4];
        let distance_metric = match metric_byte {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::Cosine,
            2 => DistanceMetric::DotProduct,
            _ => {
                return Err(VectorError::Encoding(format!(
                    "unknown distance metric: {}",
                    metric_byte
                )))
            }
        };

        Ok(Self { name, dimension, distance_metric })
    }
}

/// Metadata about a sparse embedding space.
///
/// A sparse embedding space defines the properties of sparse embeddings stored within it:
/// - The maximum dimension (vocabulary size for SPLADE, etc.)
/// - The distance metric used for similarity search
///
/// Unlike dense embedding spaces, sparse spaces don't require a fixed dimension
/// since sparse vectors only store non-zero elements.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::{EmbeddingName, SparseEmbeddingSpace};
/// use manifoldb_vector::distance::sparse::SparseDistanceMetric;
///
/// let name = EmbeddingName::new("splade_embedding").unwrap();
/// let space = SparseEmbeddingSpace::new(name, 30522, SparseDistanceMetric::DotProduct);
///
/// assert_eq!(space.max_dimension(), 30522); // BERT vocab size
/// assert_eq!(space.distance_metric(), SparseDistanceMetric::DotProduct);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseEmbeddingSpace {
    name: EmbeddingName,
    /// Maximum dimension (exclusive upper bound on indices).
    max_dimension: u32,
    distance_metric: SparseDistanceMetric,
}

impl SparseEmbeddingSpace {
    /// Create a new sparse embedding space.
    #[must_use]
    pub const fn new(
        name: EmbeddingName,
        max_dimension: u32,
        distance_metric: SparseDistanceMetric,
    ) -> Self {
        Self { name, max_dimension, distance_metric }
    }

    /// Get the name of the embedding space.
    #[must_use]
    pub fn name(&self) -> &EmbeddingName {
        &self.name
    }

    /// Get the maximum dimension (vocabulary size) for this space.
    #[must_use]
    pub fn max_dimension(&self) -> u32 {
        self.max_dimension
    }

    /// Get the distance metric used for similarity search.
    #[must_use]
    pub fn distance_metric(&self) -> SparseDistanceMetric {
        self.distance_metric
    }

    /// Encode the sparse embedding space to bytes.
    ///
    /// Format:
    /// - 1 byte: version (2 for sparse)
    /// - 2 bytes: name length (big-endian u16)
    /// - N bytes: name (UTF-8)
    /// - 4 bytes: max dimension (big-endian u32)
    /// - 1 byte: distance metric
    ///
    /// # Errors
    ///
    /// Returns an error if the name length exceeds u16::MAX.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VectorError> {
        let name_bytes = self.name.as_str().as_bytes();
        let mut bytes = Vec::with_capacity(8 + name_bytes.len());

        // Version (2 for sparse spaces)
        bytes.push(2);

        // Name length and name
        let name_len = u16::try_from(name_bytes.len()).map_err(|_| {
            VectorError::Encoding(format!(
                "embedding name too long: {} bytes exceeds maximum of {}",
                name_bytes.len(),
                u16::MAX
            ))
        })?;
        bytes.extend_from_slice(&name_len.to_be_bytes());
        bytes.extend_from_slice(name_bytes);

        // Max dimension
        bytes.extend_from_slice(&self.max_dimension.to_be_bytes());

        // Distance metric
        bytes.push(match self.distance_metric {
            SparseDistanceMetric::Euclidean => 0,
            SparseDistanceMetric::Cosine => 1,
            SparseDistanceMetric::DotProduct => 2,
        });

        Ok(bytes)
    }

    /// Decode a sparse embedding space from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are invalid or truncated.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.is_empty() {
            return Err(VectorError::Encoding("empty sparse embedding space data".to_string()));
        }

        let version = bytes[0];
        if version != 2 {
            return Err(VectorError::Encoding(format!(
                "unsupported sparse embedding space version: {} (expected 2)",
                version
            )));
        }

        if bytes.len() < 3 {
            return Err(VectorError::Encoding("truncated sparse embedding space data".to_string()));
        }

        // Name length
        let name_len = u16::from_be_bytes([bytes[1], bytes[2]]) as usize;

        if bytes.len() < 3 + name_len + 5 {
            return Err(VectorError::Encoding("truncated sparse embedding space data".to_string()));
        }

        // Name
        let name_bytes = &bytes[3..3 + name_len];
        let name_str = std::str::from_utf8(name_bytes)
            .map_err(|e| VectorError::Encoding(format!("invalid UTF-8 in name: {}", e)))?;
        let name = EmbeddingName::new(name_str)?;

        // Max dimension
        let dim_offset = 3 + name_len;
        let max_dimension = u32::from_be_bytes([
            bytes[dim_offset],
            bytes[dim_offset + 1],
            bytes[dim_offset + 2],
            bytes[dim_offset + 3],
        ]);

        // Distance metric
        let metric_byte = bytes[dim_offset + 4];
        let distance_metric = match metric_byte {
            0 => SparseDistanceMetric::Euclidean,
            1 => SparseDistanceMetric::Cosine,
            2 => SparseDistanceMetric::DotProduct,
            _ => {
                return Err(VectorError::Encoding(format!(
                    "unknown sparse distance metric: {}",
                    metric_byte
                )))
            }
        };

        Ok(Self { name, max_dimension, distance_metric })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_name_valid() {
        let name = EmbeddingName::new("text_embedding").unwrap();
        assert_eq!(name.as_str(), "text_embedding");

        let name2 = EmbeddingName::new("image-embedding-v2").unwrap();
        assert_eq!(name2.as_str(), "image-embedding-v2");

        let name3 = EmbeddingName::new("Embedding123").unwrap();
        assert_eq!(name3.as_str(), "Embedding123");
    }

    #[test]
    fn embedding_name_empty_fails() {
        let result = EmbeddingName::new("");
        assert!(result.is_err());
    }

    #[test]
    fn embedding_name_invalid_chars_fails() {
        let result = EmbeddingName::new("text embedding"); // Space not allowed
        assert!(result.is_err());

        let result = EmbeddingName::new("text.embedding"); // Dot not allowed
        assert!(result.is_err());

        let result = EmbeddingName::new("text/embedding"); // Slash not allowed
        assert!(result.is_err());
    }

    #[test]
    fn embedding_space_roundtrip() {
        let name = EmbeddingName::new("test_space").unwrap();
        let space = EmbeddingSpace::new(name, 384, DistanceMetric::Cosine);

        let bytes = space.to_bytes().unwrap();
        let restored = EmbeddingSpace::from_bytes(&bytes).unwrap();

        assert_eq!(space.name().as_str(), restored.name().as_str());
        assert_eq!(space.dimension(), restored.dimension());
        assert_eq!(space.distance_metric(), restored.distance_metric());
    }

    #[test]
    fn embedding_space_different_metrics() {
        for metric in
            [DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::DotProduct]
        {
            let name = EmbeddingName::new("test").unwrap();
            let space = EmbeddingSpace::new(name, 128, metric);

            let bytes = space.to_bytes().unwrap();
            let restored = EmbeddingSpace::from_bytes(&bytes).unwrap();

            assert_eq!(space.distance_metric(), restored.distance_metric());
        }
    }

    #[test]
    fn embedding_space_from_empty_bytes_fails() {
        let result = EmbeddingSpace::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn embedding_space_from_truncated_bytes_fails() {
        let result = EmbeddingSpace::from_bytes(&[1, 0, 5]); // Version + partial length
        assert!(result.is_err());
    }

    #[test]
    fn embedding_space_from_invalid_version_fails() {
        let result = EmbeddingSpace::from_bytes(&[99, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_embedding_space_roundtrip() {
        let name = EmbeddingName::new("splade_space").unwrap();
        let space = SparseEmbeddingSpace::new(name, 30522, SparseDistanceMetric::DotProduct);

        let bytes = space.to_bytes().unwrap();
        let restored = SparseEmbeddingSpace::from_bytes(&bytes).unwrap();

        assert_eq!(space.name().as_str(), restored.name().as_str());
        assert_eq!(space.max_dimension(), restored.max_dimension());
        assert_eq!(space.distance_metric(), restored.distance_metric());
    }

    #[test]
    fn sparse_embedding_space_different_metrics() {
        for metric in [
            SparseDistanceMetric::Euclidean,
            SparseDistanceMetric::Cosine,
            SparseDistanceMetric::DotProduct,
        ] {
            let name = EmbeddingName::new("test").unwrap();
            let space = SparseEmbeddingSpace::new(name, 10000, metric);

            let bytes = space.to_bytes().unwrap();
            let restored = SparseEmbeddingSpace::from_bytes(&bytes).unwrap();

            assert_eq!(space.distance_metric(), restored.distance_metric());
        }
    }

    #[test]
    fn sparse_embedding_space_from_empty_bytes_fails() {
        let result = SparseEmbeddingSpace::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_embedding_space_from_wrong_version_fails() {
        // Version 1 is for dense spaces, not sparse
        let result = SparseEmbeddingSpace::from_bytes(&[1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(result.is_err());
    }
}
