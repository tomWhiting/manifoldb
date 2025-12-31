//! Types for backup and restore operations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The current backup format version.
pub const BACKUP_FORMAT_VERSION: u32 = 1;

/// Backup format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackupFormat {
    /// JSON-lines format (one JSON object per line).
    JsonLines,
}

impl Default for BackupFormat {
    fn default() -> Self {
        Self::JsonLines
    }
}

/// Metadata about a backup file.
///
/// This is always the first record in a backup file and contains
/// information needed to verify and restore the backup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// The backup format version.
    pub version: u32,

    /// The backup format type.
    pub format: BackupFormat,

    /// The timestamp when the backup was created (Unix epoch seconds).
    pub created_at: u64,

    /// The database sequence number at backup time.
    ///
    /// This is used for incremental backups to determine what has
    /// changed since the last backup.
    pub sequence_number: u64,

    /// Whether this is a full or incremental backup.
    pub is_incremental: bool,

    /// The sequence number of the previous backup (for incremental backups).
    pub previous_sequence: Option<u64>,

    /// Statistics about the backup contents.
    pub statistics: BackupStatistics,

    /// Optional user-provided description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Additional metadata fields for extensibility.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, String>,
}

impl BackupMetadata {
    /// Create new backup metadata for a full backup.
    pub fn new_full(sequence_number: u64) -> Self {
        Self {
            version: BACKUP_FORMAT_VERSION,
            format: BackupFormat::default(),
            created_at: current_timestamp(),
            sequence_number,
            is_incremental: false,
            previous_sequence: None,
            statistics: BackupStatistics::default(),
            description: None,
            extra: HashMap::new(),
        }
    }

    /// Create new backup metadata for an incremental backup.
    pub fn new_incremental(sequence_number: u64, previous_sequence: u64) -> Self {
        Self {
            version: BACKUP_FORMAT_VERSION,
            format: BackupFormat::default(),
            created_at: current_timestamp(),
            sequence_number,
            is_incremental: true,
            previous_sequence: Some(previous_sequence),
            statistics: BackupStatistics::default(),
            description: None,
            extra: HashMap::new(),
        }
    }

    /// Set the description for this backup.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add extra metadata.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Statistics about backup contents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackupStatistics {
    /// Number of entities in the backup.
    pub entity_count: u64,

    /// Number of edges in the backup.
    pub edge_count: u64,

    /// Number of metadata entries in the backup.
    pub metadata_count: u64,

    /// Total number of records (all types).
    pub total_records: u64,

    /// Size in bytes of the uncompressed backup data.
    pub uncompressed_size: u64,
}

impl BackupStatistics {
    /// Increment the entity count.
    pub fn add_entity(&mut self) {
        self.entity_count += 1;
        self.total_records += 1;
    }

    /// Increment the edge count.
    pub fn add_edge(&mut self) {
        self.edge_count += 1;
        self.total_records += 1;
    }

    /// Increment the metadata count.
    pub fn add_metadata(&mut self) {
        self.metadata_count += 1;
        self.total_records += 1;
    }

    /// Add to the uncompressed size.
    pub fn add_size(&mut self, size: u64) {
        self.uncompressed_size += size;
    }
}

/// The type of a backup record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackupRecordType {
    /// Backup metadata (always first record).
    Metadata,
    /// An entity/node record.
    Entity,
    /// An edge/relationship record.
    Edge,
    /// A raw key-value metadata record.
    KeyValue,
    /// End-of-backup marker (optional, for verification).
    EndOfBackup,
}

/// A single record in a backup file.
///
/// Each line in a JSON-lines backup represents one record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecord {
    /// The type of this record.
    #[serde(rename = "type")]
    pub record_type: BackupRecordType,

    /// The record data.
    pub data: RecordData,

    /// Optional table name for key-value records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
}

impl BackupRecord {
    /// Create a metadata record.
    pub fn metadata(meta: BackupMetadata) -> Self {
        Self {
            record_type: BackupRecordType::Metadata,
            data: RecordData::Metadata(meta),
            table: None,
        }
    }

    /// Create an entity record.
    pub fn entity(entity: EntityRecord) -> Self {
        Self {
            record_type: BackupRecordType::Entity,
            data: RecordData::Entity(entity),
            table: None,
        }
    }

    /// Create an edge record.
    pub fn edge(edge: EdgeRecord) -> Self {
        Self { record_type: BackupRecordType::Edge, data: RecordData::Edge(edge), table: None }
    }

    /// Create a key-value record.
    pub fn key_value(table: String, key: Vec<u8>, value: Vec<u8>) -> Self {
        Self {
            record_type: BackupRecordType::KeyValue,
            data: RecordData::KeyValue(KeyValueRecord { key, value }),
            table: Some(table),
        }
    }

    /// Create an end-of-backup marker.
    pub fn end_of_backup(stats: BackupStatistics) -> Self {
        Self {
            record_type: BackupRecordType::EndOfBackup,
            data: RecordData::EndOfBackup(stats),
            table: None,
        }
    }
}

/// The data payload of a backup record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RecordData {
    /// Backup metadata.
    Metadata(BackupMetadata),
    /// An entity record.
    Entity(EntityRecord),
    /// An edge record.
    Edge(EdgeRecord),
    /// A raw key-value record.
    KeyValue(KeyValueRecord),
    /// End-of-backup statistics.
    EndOfBackup(BackupStatistics),
}

/// A serialized entity for backup.
///
/// Uses the same structure as the core Entity type but with
/// explicit serde annotations for portability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// The entity ID.
    pub id: u64,
    /// The entity labels.
    pub labels: Vec<String>,
    /// The entity properties.
    pub properties: HashMap<String, serde_json::Value>,
}

impl EntityRecord {
    /// Create an entity record from a core Entity.
    pub fn from_entity(entity: &manifoldb_core::Entity) -> Self {
        let properties =
            entity.properties.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect();

        Self {
            id: entity.id.as_u64(),
            labels: entity.labels.iter().map(|l| l.as_str().to_owned()).collect(),
            properties,
        }
    }

    /// Convert to a core Entity.
    pub fn to_entity(&self) -> manifoldb_core::Entity {
        let mut entity = manifoldb_core::Entity::new(manifoldb_core::EntityId::new(self.id));

        for label in &self.labels {
            entity = entity.with_label(label.as_str());
        }

        for (key, value) in &self.properties {
            entity = entity.with_property(key.as_str(), json_to_value(value));
        }

        entity
    }
}

/// A serialized edge for backup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRecord {
    /// The edge ID.
    pub id: u64,
    /// The source entity ID.
    pub source: u64,
    /// The target entity ID.
    pub target: u64,
    /// The edge type.
    pub edge_type: String,
    /// The edge properties.
    pub properties: HashMap<String, serde_json::Value>,
}

impl EdgeRecord {
    /// Create an edge record from a core Edge.
    pub fn from_edge(edge: &manifoldb_core::Edge) -> Self {
        let properties =
            edge.properties.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect();

        Self {
            id: edge.id.as_u64(),
            source: edge.source.as_u64(),
            target: edge.target.as_u64(),
            edge_type: edge.edge_type.as_str().to_owned(),
            properties,
        }
    }

    /// Convert to a core Edge.
    pub fn to_edge(&self) -> manifoldb_core::Edge {
        let mut edge = manifoldb_core::Edge::new(
            manifoldb_core::EdgeId::new(self.id),
            manifoldb_core::EntityId::new(self.source),
            manifoldb_core::EntityId::new(self.target),
            self.edge_type.as_str(),
        );

        for (key, value) in &self.properties {
            edge = edge.with_property(key.as_str(), json_to_value(value));
        }

        edge
    }
}

/// A raw key-value record for backup.
///
/// Used for metadata and other non-entity/edge data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValueRecord {
    /// The key (base64-encoded bytes).
    #[serde(with = "base64_bytes")]
    pub key: Vec<u8>,
    /// The value (base64-encoded bytes).
    #[serde(with = "base64_bytes")]
    pub value: Vec<u8>,
}

/// Convert a ManifoldDB Value to a JSON value for serialization.
fn value_to_json(value: &manifoldb_core::Value) -> serde_json::Value {
    match value {
        manifoldb_core::Value::Null => serde_json::Value::Null,
        manifoldb_core::Value::Bool(b) => serde_json::Value::Bool(*b),
        manifoldb_core::Value::Int(i) => serde_json::Value::Number((*i).into()),
        manifoldb_core::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        manifoldb_core::Value::String(s) => serde_json::Value::String(s.clone()),
        manifoldb_core::Value::Bytes(b) => {
            use base64::Engine;
            serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        manifoldb_core::Value::Vector(v) => serde_json::Value::Array(
            v.iter()
                .map(|f| {
                    serde_json::Number::from_f64(f64::from(*f))
                        .map_or(serde_json::Value::Null, serde_json::Value::Number)
                })
                .collect(),
        ),
        manifoldb_core::Value::SparseVector(v) => {
            // Serialize sparse vector as array of [index, value] pairs
            serde_json::Value::Array(
                v.iter()
                    .map(|(idx, val)| {
                        serde_json::Value::Array(vec![
                            serde_json::Value::Number((*idx).into()),
                            serde_json::Number::from_f64(f64::from(*val))
                                .map_or(serde_json::Value::Null, serde_json::Value::Number),
                        ])
                    })
                    .collect(),
            )
        }
        manifoldb_core::Value::MultiVector(vecs) => {
            // Serialize multi-vector as array of token embedding arrays
            serde_json::Value::Array(
                vecs.iter()
                    .map(|v| {
                        serde_json::Value::Array(
                            v.iter()
                                .map(|f| {
                                    serde_json::Number::from_f64(f64::from(*f))
                                        .map_or(serde_json::Value::Null, serde_json::Value::Number)
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            )
        }
        manifoldb_core::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(value_to_json).collect())
        }
    }
}

/// Convert a JSON value back to a ManifoldDB Value.
fn json_to_value(json: &serde_json::Value) -> manifoldb_core::Value {
    match json {
        serde_json::Value::Null => manifoldb_core::Value::Null,
        serde_json::Value::Bool(b) => manifoldb_core::Value::Bool(*b),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(manifoldb_core::Value::Int)
            .or_else(|| n.as_f64().map(manifoldb_core::Value::Float))
            .unwrap_or(manifoldb_core::Value::Null),
        serde_json::Value::String(s) => manifoldb_core::Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            // Try to detect if this is a vector (all f64) or a general array
            let as_floats: Option<Vec<f32>> =
                arr.iter().map(|v| v.as_f64().map(|f| f as f32)).collect();

            if let Some(floats) = as_floats {
                manifoldb_core::Value::Vector(floats)
            } else {
                manifoldb_core::Value::Array(arr.iter().map(json_to_value).collect())
            }
        }
        serde_json::Value::Object(_) => {
            // Objects are not directly supported in ManifoldDB Value
            manifoldb_core::Value::Null
        }
    }
}

/// Get the current Unix timestamp in seconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Custom serde module for base64-encoded bytes.
mod base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        serializer.serialize_str(&encoded)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        base64::engine::general_purpose::STANDARD.decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_metadata_full() {
        let meta = BackupMetadata::new_full(42);
        assert_eq!(meta.version, BACKUP_FORMAT_VERSION);
        assert!(!meta.is_incremental);
        assert_eq!(meta.sequence_number, 42);
        assert!(meta.previous_sequence.is_none());
    }

    #[test]
    fn test_backup_metadata_incremental() {
        let meta = BackupMetadata::new_incremental(100, 50);
        assert!(meta.is_incremental);
        assert_eq!(meta.sequence_number, 100);
        assert_eq!(meta.previous_sequence, Some(50));
    }

    #[test]
    fn test_entity_record_roundtrip() {
        let entity = manifoldb_core::Entity::new(manifoldb_core::EntityId::new(123))
            .with_label("Person")
            .with_property("name", "Alice")
            .with_property("age", 30i64);

        let record = EntityRecord::from_entity(&entity);
        let restored = record.to_entity();

        assert_eq!(restored.id.as_u64(), 123);
        assert!(restored.has_label("Person"));
        assert_eq!(
            restored.properties.get("name"),
            Some(&manifoldb_core::Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_edge_record_roundtrip() {
        let edge = manifoldb_core::Edge::new(
            manifoldb_core::EdgeId::new(456),
            manifoldb_core::EntityId::new(1),
            manifoldb_core::EntityId::new(2),
            "FOLLOWS",
        )
        .with_property("since", 2024i64);

        let record = EdgeRecord::from_edge(&edge);
        let restored = record.to_edge();

        assert_eq!(restored.id.as_u64(), 456);
        assert_eq!(restored.source.as_u64(), 1);
        assert_eq!(restored.target.as_u64(), 2);
        assert_eq!(restored.edge_type.as_str(), "FOLLOWS");
    }

    #[test]
    fn test_value_json_roundtrip() {
        let values = vec![
            manifoldb_core::Value::Null,
            manifoldb_core::Value::Bool(true),
            manifoldb_core::Value::Int(42),
            manifoldb_core::Value::Float(3.14),
            manifoldb_core::Value::String("hello".to_string()),
        ];

        for value in values {
            let json = value_to_json(&value);
            let restored = json_to_value(&json);

            match (&value, &restored) {
                (manifoldb_core::Value::Float(a), manifoldb_core::Value::Float(b)) => {
                    assert!((a - b).abs() < f64::EPSILON);
                }
                _ => assert_eq!(value, restored),
            }
        }
    }
}
