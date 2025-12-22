//! Point types for vector collections.
//!
//! This module provides types for Qdrant-style vector collections where points
//! can have multiple named vectors (dense, sparse, or multi-vector).

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::VectorError;

/// A validated collection name.
///
/// Collection names identify different vector collections.
/// Names must be non-empty and contain only alphanumeric characters, underscores,
/// and hyphens.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::CollectionName;
///
/// let name = CollectionName::new("documents").unwrap();
/// assert_eq!(name.as_str(), "documents");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionName(String);

impl CollectionName {
    /// Create a new collection name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is empty or contains invalid characters.
    pub fn new(name: impl Into<String>) -> Result<Self, VectorError> {
        let name = name.into();

        if name.is_empty() {
            return Err(VectorError::InvalidName("collection name cannot be empty".to_string()));
        }

        // Validate characters: alphanumeric, underscore, hyphen
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(VectorError::InvalidName(format!(
                "collection name '{}' contains invalid characters (allowed: alphanumeric, underscore, hyphen)",
                name
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

impl fmt::Display for CollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for CollectionName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A validated vector name.
///
/// Vector names identify different vectors within a point.
/// Names must be non-empty and contain only alphanumeric characters, underscores,
/// and hyphens.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::VectorName;
///
/// let name = VectorName::new("text_embedding").unwrap();
/// assert_eq!(name.as_str(), "text_embedding");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorName(String);

impl VectorName {
    /// Create a new vector name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is empty or contains invalid characters.
    pub fn new(name: impl Into<String>) -> Result<Self, VectorError> {
        let name = name.into();

        if name.is_empty() {
            return Err(VectorError::InvalidName("vector name cannot be empty".to_string()));
        }

        // Validate characters: alphanumeric, underscore, hyphen
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(VectorError::InvalidName(format!(
                "vector name '{}' contains invalid characters (allowed: alphanumeric, underscore, hyphen)",
                name
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

impl fmt::Display for VectorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for VectorName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Type of vector in a collection schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorType {
    /// Dense vector (f32 array).
    Dense,
    /// Sparse vector (index-value pairs).
    Sparse,
    /// Multi-vector (array of f32 arrays, e.g., ColBERT).
    Multi,
}

/// Configuration for a vector in a collection schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Type of vector.
    pub vector_type: VectorType,
    /// Dimension of the vector (for dense and multi-vectors).
    /// For sparse vectors, this is the max dimension (vocabulary size).
    pub dimension: u32,
}

impl VectorConfig {
    /// Create a new dense vector config.
    #[must_use]
    pub const fn dense(dimension: u32) -> Self {
        Self { vector_type: VectorType::Dense, dimension }
    }

    /// Create a new sparse vector config.
    #[must_use]
    pub const fn sparse(max_dimension: u32) -> Self {
        Self { vector_type: VectorType::Sparse, dimension: max_dimension }
    }

    /// Create a new multi-vector config.
    #[must_use]
    pub const fn multi(dimension: u32) -> Self {
        Self { vector_type: VectorType::Multi, dimension }
    }
}

/// Schema for a collection defining which vectors are expected.
///
/// A collection schema defines the named vectors that points in the collection
/// should have. Each vector has a name and configuration (type, dimension).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchema {
    /// Vector configurations by name.
    vectors: HashMap<String, VectorConfig>,
}

impl CollectionSchema {
    /// Create a new empty collection schema.
    #[must_use]
    pub fn new() -> Self {
        Self { vectors: HashMap::new() }
    }

    /// Add a vector to the schema.
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, config: VectorConfig) -> Self {
        self.vectors.insert(name.into(), config);
        self
    }

    /// Get the configuration for a vector by name.
    #[must_use]
    pub fn get_vector(&self, name: &str) -> Option<&VectorConfig> {
        self.vectors.get(name)
    }

    /// Get all vector configurations.
    #[must_use]
    pub fn vectors(&self) -> &HashMap<String, VectorConfig> {
        &self.vectors
    }

    /// Check if the schema is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the number of vectors in the schema.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
}

impl Default for CollectionSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    /// The name of the collection.
    name: CollectionName,
    /// The schema defining vectors in this collection.
    schema: CollectionSchema,
}

impl Collection {
    /// Create a new collection.
    #[must_use]
    pub fn new(name: CollectionName, schema: CollectionSchema) -> Self {
        Self { name, schema }
    }

    /// Get the collection name.
    #[must_use]
    pub fn name(&self) -> &CollectionName {
        &self.name
    }

    /// Get the collection schema.
    #[must_use]
    pub fn schema(&self) -> &CollectionSchema {
        &self.schema
    }

    /// Encode the collection metadata to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VectorError> {
        serde_json::to_vec(self)
            .map_err(|e| VectorError::Encoding(format!("failed to encode collection: {}", e)))
    }

    /// Decode collection metadata from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        serde_json::from_slice(bytes)
            .map_err(|e| VectorError::Encoding(format!("failed to decode collection: {}", e)))
    }
}

/// A point's payload (JSON data).
///
/// The payload stores arbitrary JSON data associated with a point.
// Note: We cannot derive Eq because serde_json::Value doesn't implement Eq
// (it contains floating point numbers which don't have a total ordering)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct Payload(JsonValue);

impl Payload {
    /// Create a new empty payload.
    #[must_use]
    pub fn new() -> Self {
        Self(JsonValue::Object(serde_json::Map::new()))
    }

    /// Create a payload from a JSON value.
    #[must_use]
    pub fn from_value(value: JsonValue) -> Self {
        Self(value)
    }

    /// Get the underlying JSON value.
    #[must_use]
    pub fn value(&self) -> &JsonValue {
        &self.0
    }

    /// Get a mutable reference to the underlying JSON value.
    pub fn value_mut(&mut self) -> &mut JsonValue {
        &mut self.0
    }

    /// Consume and return the underlying JSON value.
    #[must_use]
    pub fn into_value(self) -> JsonValue {
        self.0
    }

    /// Check if the payload is empty (null or empty object).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.0 {
            JsonValue::Null => true,
            JsonValue::Object(m) => m.is_empty(),
            _ => false,
        }
    }

    /// Get a field from the payload.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key)
    }

    /// Insert a field into the payload.
    ///
    /// The payload must be an object for this to work.
    pub fn insert(&mut self, key: impl Into<String>, value: JsonValue) {
        if let JsonValue::Object(ref mut map) = self.0 {
            map.insert(key.into(), value);
        }
    }

    /// Encode the payload to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VectorError> {
        serde_json::to_vec(&self.0)
            .map_err(|e| VectorError::Encoding(format!("failed to encode payload: {}", e)))
    }

    /// Decode a payload from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        let value = serde_json::from_slice(bytes)
            .map_err(|e| VectorError::Encoding(format!("failed to decode payload: {}", e)))?;
        Ok(Self(value))
    }
}

impl Default for Payload {
    fn default() -> Self {
        Self::new()
    }
}

impl From<JsonValue> for Payload {
    fn from(value: JsonValue) -> Self {
        Self::from_value(value)
    }
}

/// A named vector value that can be dense, sparse, or multi-vector.
#[derive(Debug, Clone, PartialEq)]
pub enum NamedVector {
    /// Dense vector (f32 array).
    Dense(Vec<f32>),
    /// Sparse vector (sorted index-value pairs).
    Sparse(Vec<(u32, f32)>),
    /// Multi-vector (array of f32 arrays).
    Multi(Vec<Vec<f32>>),
}

impl NamedVector {
    /// Get the vector type.
    #[must_use]
    pub fn vector_type(&self) -> VectorType {
        match self {
            Self::Dense(_) => VectorType::Dense,
            Self::Sparse(_) => VectorType::Sparse,
            Self::Multi(_) => VectorType::Multi,
        }
    }

    /// Get as a dense vector.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(v) => Some(v),
            _ => None,
        }
    }

    /// Get as a sparse vector.
    #[must_use]
    pub fn as_sparse(&self) -> Option<&[(u32, f32)]> {
        match self {
            Self::Sparse(v) => Some(v),
            _ => None,
        }
    }

    /// Get as a multi-vector.
    #[must_use]
    pub fn as_multi(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::Multi(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_name_valid() {
        let name = CollectionName::new("documents").unwrap();
        assert_eq!(name.as_str(), "documents");

        let name2 = CollectionName::new("my-collection_v2").unwrap();
        assert_eq!(name2.as_str(), "my-collection_v2");
    }

    #[test]
    fn collection_name_empty_fails() {
        assert!(CollectionName::new("").is_err());
    }

    #[test]
    fn collection_name_invalid_chars_fails() {
        assert!(CollectionName::new("my collection").is_err()); // space
        assert!(CollectionName::new("my.collection").is_err()); // dot
        assert!(CollectionName::new("my/collection").is_err()); // slash
    }

    #[test]
    fn vector_name_valid() {
        let name = VectorName::new("text_embedding").unwrap();
        assert_eq!(name.as_str(), "text_embedding");
    }

    #[test]
    fn vector_name_empty_fails() {
        assert!(VectorName::new("").is_err());
    }

    #[test]
    fn collection_schema_builder() {
        let schema = CollectionSchema::new()
            .with_vector("dense", VectorConfig::dense(384))
            .with_vector("sparse", VectorConfig::sparse(30522))
            .with_vector("colbert", VectorConfig::multi(128));

        assert_eq!(schema.len(), 3);

        let dense = schema.get_vector("dense").unwrap();
        assert_eq!(dense.vector_type, VectorType::Dense);
        assert_eq!(dense.dimension, 384);

        let sparse = schema.get_vector("sparse").unwrap();
        assert_eq!(sparse.vector_type, VectorType::Sparse);
        assert_eq!(sparse.dimension, 30522);

        let multi = schema.get_vector("colbert").unwrap();
        assert_eq!(multi.vector_type, VectorType::Multi);
        assert_eq!(multi.dimension, 128);
    }

    #[test]
    fn collection_roundtrip() {
        let name = CollectionName::new("test").unwrap();
        let schema = CollectionSchema::new().with_vector("embedding", VectorConfig::dense(768));
        let collection = Collection::new(name, schema);

        let bytes = collection.to_bytes().unwrap();
        let restored = Collection::from_bytes(&bytes).unwrap();

        assert_eq!(collection.name().as_str(), restored.name().as_str());
        assert_eq!(collection.schema().len(), restored.schema().len());
    }

    #[test]
    fn payload_operations() {
        let mut payload = Payload::new();
        assert!(payload.is_empty());

        payload.insert("title", JsonValue::String("Hello".to_string()));
        payload.insert("count", JsonValue::Number(42.into()));

        assert!(!payload.is_empty());
        assert_eq!(payload.get("title"), Some(&JsonValue::String("Hello".to_string())));
        assert_eq!(payload.get("count"), Some(&JsonValue::Number(42.into())));
    }

    #[test]
    fn payload_roundtrip() {
        let mut payload = Payload::new();
        payload.insert("title", JsonValue::String("Test".to_string()));

        let bytes = payload.to_bytes().unwrap();
        let restored = Payload::from_bytes(&bytes).unwrap();

        assert_eq!(payload.value(), restored.value());
    }

    #[test]
    fn named_vector_types() {
        let dense = NamedVector::Dense(vec![0.1, 0.2, 0.3]);
        assert_eq!(dense.vector_type(), VectorType::Dense);
        assert!(dense.as_dense().is_some());
        assert!(dense.as_sparse().is_none());
        assert!(dense.as_multi().is_none());

        let sparse = NamedVector::Sparse(vec![(0, 0.5), (10, 0.3)]);
        assert_eq!(sparse.vector_type(), VectorType::Sparse);
        assert!(sparse.as_sparse().is_some());
        assert!(sparse.as_dense().is_none());

        let multi = NamedVector::Multi(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert_eq!(multi.vector_type(), VectorType::Multi);
        assert!(multi.as_multi().is_some());
        assert!(multi.as_dense().is_none());
    }
}
