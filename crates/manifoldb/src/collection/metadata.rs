//! Collection metadata types.
//!
//! This module defines the `Collection` struct and related types for
//! representing collection metadata.

use std::collections::HashMap;
use std::fmt;

use manifoldb_core::CollectionId;
use serde::{Deserialize, Serialize};

use super::VectorConfig;

/// A validated collection name.
///
/// Collection names must be non-empty and contain only alphanumeric characters,
/// underscores, and hyphens. This ensures compatibility with storage backends
/// and prevents injection attacks in queries.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::CollectionName;
///
/// let name = CollectionName::new("my_documents").unwrap();
/// assert_eq!(name.as_str(), "my_documents");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionName(String);

impl CollectionName {
    /// Create a new validated collection name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is empty or contains invalid characters.
    pub fn new(name: impl Into<String>) -> Result<Self, CollectionNameError> {
        let name = name.into();

        if name.is_empty() {
            return Err(CollectionNameError::Empty);
        }

        // Validate characters: alphanumeric, underscore, hyphen
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(CollectionNameError::InvalidCharacters(name));
        }

        // Limit length to prevent abuse
        if name.len() > 255 {
            return Err(CollectionNameError::TooLong(name.len()));
        }

        Ok(Self(name))
    }

    /// Get the name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
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

/// Errors that can occur when creating a collection name.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CollectionNameError {
    #[error("collection name cannot be empty")]
    Empty,

    #[error("collection name '{0}' contains invalid characters (allowed: alphanumeric, underscore, hyphen)")]
    InvalidCharacters(String),

    #[error("collection name too long: {0} bytes (maximum: 255)")]
    TooLong(usize),
}

/// Metadata for a collection of entities with named vectors.
///
/// A collection is a logical grouping of entities that share the same
/// vector schema. Each collection can have multiple named vectors with
/// different configurations.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::{Collection, VectorConfig};
/// use manifoldb_vector::distance::DistanceMetric;
///
/// let collection = Collection::new("documents")
///     .with_vector("dense", VectorConfig::dense(768, DistanceMetric::Cosine))
///     .with_vector("sparse", VectorConfig::sparse(30522));
///
/// assert_eq!(collection.name().as_str(), "documents");
/// assert_eq!(collection.vectors().len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Collection {
    /// Unique identifier for the collection.
    id: CollectionId,
    /// The collection name.
    name: CollectionName,
    /// Named vector configurations.
    vectors: HashMap<String, VectorConfig>,
    /// Optional payload schema for validation.
    payload_schema: Option<PayloadSchema>,
    /// Creation timestamp (Unix timestamp in seconds).
    created_at: u64,
    /// Last modification timestamp.
    updated_at: u64,
}

impl Collection {
    /// Create a new collection with the given name.
    ///
    /// The collection starts with no vectors configured. Use `with_vector`
    /// to add named vector configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid.
    pub fn new(id: CollectionId, name: CollectionName) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id,
            name,
            vectors: HashMap::new(),
            payload_schema: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a named vector configuration.
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, config: VectorConfig) -> Self {
        self.vectors.insert(name.into(), config);
        self
    }

    /// Add multiple vector configurations.
    #[must_use]
    pub fn with_vectors(mut self, vectors: HashMap<String, VectorConfig>) -> Self {
        self.vectors.extend(vectors);
        self
    }

    /// Set the payload schema.
    #[must_use]
    pub fn with_payload_schema(mut self, schema: PayloadSchema) -> Self {
        self.payload_schema = Some(schema);
        self
    }

    /// Get the collection ID.
    #[must_use]
    pub fn id(&self) -> CollectionId {
        self.id
    }

    /// Get the collection name.
    #[must_use]
    pub fn name(&self) -> &CollectionName {
        &self.name
    }

    /// Get the named vector configurations.
    #[must_use]
    pub fn vectors(&self) -> &HashMap<String, VectorConfig> {
        &self.vectors
    }

    /// Get a specific vector configuration by name.
    #[must_use]
    pub fn get_vector(&self, name: &str) -> Option<&VectorConfig> {
        self.vectors.get(name)
    }

    /// Check if a named vector exists.
    #[must_use]
    pub fn has_vector(&self, name: &str) -> bool {
        self.vectors.contains_key(name)
    }

    /// Get the payload schema.
    #[must_use]
    pub fn payload_schema(&self) -> Option<&PayloadSchema> {
        self.payload_schema.as_ref()
    }

    /// Get the creation timestamp.
    #[must_use]
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Get the last modification timestamp.
    #[must_use]
    pub fn updated_at(&self) -> u64 {
        self.updated_at
    }

    /// Add a named vector configuration (mutable).
    pub fn add_vector(&mut self, name: impl Into<String>, config: VectorConfig) {
        self.vectors.insert(name.into(), config);
        self.touch();
    }

    /// Remove a named vector configuration.
    pub fn remove_vector(&mut self, name: &str) -> Option<VectorConfig> {
        let config = self.vectors.remove(name);
        if config.is_some() {
            self.touch();
        }
        config
    }

    /// Update the modification timestamp to now.
    fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }
}

/// Schema for validating entity payloads in a collection.
///
/// Payload schemas define the expected fields and types for entity properties.
/// This is optional but helps ensure data consistency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayloadSchema {
    /// Field definitions.
    pub fields: HashMap<String, PayloadFieldType>,
    /// Whether to allow fields not defined in the schema.
    pub allow_extra_fields: bool,
}

impl PayloadSchema {
    /// Create a new empty schema that allows extra fields.
    #[must_use]
    pub fn new() -> Self {
        Self { fields: HashMap::new(), allow_extra_fields: true }
    }

    /// Create a strict schema that rejects extra fields.
    #[must_use]
    pub fn strict() -> Self {
        Self { fields: HashMap::new(), allow_extra_fields: false }
    }

    /// Add a field definition.
    #[must_use]
    pub fn with_field(mut self, name: impl Into<String>, field_type: PayloadFieldType) -> Self {
        self.fields.insert(name.into(), field_type);
        self
    }

    /// Set whether to allow extra fields.
    #[must_use]
    pub fn with_extra_fields(mut self, allow: bool) -> Self {
        self.allow_extra_fields = allow;
        self
    }
}

impl Default for PayloadSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of a payload field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PayloadFieldType {
    /// String field.
    String { max_length: Option<usize> },
    /// Integer field.
    Integer { min: Option<i64>, max: Option<i64> },
    /// Float field.
    Float { min: Option<f64>, max: Option<f64> },
    /// Boolean field.
    Boolean,
    /// Array of values.
    Array { element_type: Box<PayloadFieldType>, max_length: Option<usize> },
    /// Nested object.
    Object { schema: Box<PayloadSchema> },
    /// Any JSON value.
    Any,
}

impl PayloadFieldType {
    /// Create a string field type.
    #[must_use]
    pub const fn string() -> Self {
        Self::String { max_length: None }
    }

    /// Create a string field with max length.
    #[must_use]
    pub const fn string_with_max_length(max: usize) -> Self {
        Self::String { max_length: Some(max) }
    }

    /// Create an integer field type.
    #[must_use]
    pub const fn integer() -> Self {
        Self::Integer { min: None, max: None }
    }

    /// Create an integer field with range.
    #[must_use]
    pub const fn integer_range(min: i64, max: i64) -> Self {
        Self::Integer { min: Some(min), max: Some(max) }
    }

    /// Create a float field type.
    #[must_use]
    pub const fn float() -> Self {
        Self::Float { min: None, max: None }
    }

    /// Create a boolean field type.
    #[must_use]
    pub const fn boolean() -> Self {
        Self::Boolean
    }

    /// Create an array field type.
    #[must_use]
    pub fn array(element_type: PayloadFieldType) -> Self {
        Self::Array { element_type: Box::new(element_type), max_length: None }
    }

    /// Create an any-type field.
    #[must_use]
    pub const fn any() -> Self {
        Self::Any
    }
}

#[cfg(test)]
mod tests {
    use manifoldb_vector::distance::DistanceMetric;

    use super::*;

    #[test]
    fn test_collection_name_valid() {
        let name = CollectionName::new("my_documents").unwrap();
        assert_eq!(name.as_str(), "my_documents");

        let name2 = CollectionName::new("docs-v2").unwrap();
        assert_eq!(name2.as_str(), "docs-v2");

        let name3 = CollectionName::new("Collection123").unwrap();
        assert_eq!(name3.as_str(), "Collection123");
    }

    #[test]
    fn test_collection_name_empty_fails() {
        let result = CollectionName::new("");
        assert!(matches!(result, Err(CollectionNameError::Empty)));
    }

    #[test]
    fn test_collection_name_invalid_chars_fails() {
        let result = CollectionName::new("my documents"); // Space
        assert!(matches!(result, Err(CollectionNameError::InvalidCharacters(_))));

        let result = CollectionName::new("my.documents"); // Dot
        assert!(matches!(result, Err(CollectionNameError::InvalidCharacters(_))));

        let result = CollectionName::new("my/documents"); // Slash
        assert!(matches!(result, Err(CollectionNameError::InvalidCharacters(_))));
    }

    #[test]
    fn test_collection_builder() {
        use crate::collection::VectorConfig;

        let collection =
            Collection::new(CollectionId::new(1), CollectionName::new("documents").unwrap())
                .with_vector("dense", VectorConfig::dense(768, DistanceMetric::Cosine))
                .with_vector("sparse", VectorConfig::sparse(30522));

        assert_eq!(collection.name().as_str(), "documents");
        assert_eq!(collection.vectors().len(), 2);
        assert!(collection.has_vector("dense"));
        assert!(collection.has_vector("sparse"));
        assert!(!collection.has_vector("nonexistent"));
    }

    #[test]
    fn test_collection_add_remove_vector() {
        use crate::collection::VectorConfig;

        let mut collection =
            Collection::new(CollectionId::new(1), CollectionName::new("test").unwrap());

        collection.add_vector("dense", VectorConfig::dense(768, DistanceMetric::Cosine));
        assert!(collection.has_vector("dense"));

        let removed = collection.remove_vector("dense");
        assert!(removed.is_some());
        assert!(!collection.has_vector("dense"));
    }

    #[test]
    fn test_payload_schema() {
        let schema = PayloadSchema::new()
            .with_field("title", PayloadFieldType::string())
            .with_field("score", PayloadFieldType::float())
            .with_field("tags", PayloadFieldType::array(PayloadFieldType::string()));

        assert_eq!(schema.fields.len(), 3);
        assert!(schema.allow_extra_fields);
    }

    #[test]
    fn test_payload_schema_strict() {
        let schema =
            PayloadSchema::strict().with_field("required_field", PayloadFieldType::string());

        assert!(!schema.allow_extra_fields);
    }
}
