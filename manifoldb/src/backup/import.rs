//! Backup import (restore) functionality.
//!
//! This module provides functions to restore database contents from a
//! JSON-lines backup file.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read};

use super::error::{BackupError, BackupResult};
use super::types::{
    BackupMetadata, BackupRecord, BackupRecordType, BackupStatistics, KeyValueRecord, RecordData,
    BACKUP_FORMAT_VERSION,
};
use crate::Database;

/// Well-known table names for restoration.
mod tables {
    pub const METADATA: &str = "metadata";
}

/// Options for controlling the import process.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ImportOptions {
    /// Whether to verify referential integrity (edges reference existing entities).
    pub verify_references: bool,

    /// Whether to skip duplicate records instead of failing.
    pub skip_duplicates: bool,

    /// Whether to rebuild indexes after import.
    pub rebuild_indexes: bool,

    /// Maximum number of records to import in a single transaction.
    /// Use None for unlimited (all in one transaction).
    pub batch_size: Option<usize>,

    /// Whether to perform a dry run (verify without writing).
    pub dry_run: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            verify_references: true,
            skip_duplicates: false,
            rebuild_indexes: true,
            batch_size: Some(10_000),
            dry_run: false,
        }
    }
}

impl ImportOptions {
    /// Create options for a fast import without verification.
    pub fn fast() -> Self {
        Self {
            verify_references: false,
            skip_duplicates: true,
            rebuild_indexes: false,
            batch_size: Some(50_000),
            dry_run: false,
        }
    }

    /// Create options for a dry run.
    pub fn dry_run() -> Self {
        Self { dry_run: true, ..Default::default() }
    }
}

/// An importer for restoring backup data.
///
/// This struct provides a streaming interface for reading and restoring
/// backup records, allowing large backups to be restored efficiently.
pub struct Importer<R: Read> {
    reader: BufReader<R>,
    metadata: Option<BackupMetadata>,
    options: ImportOptions,
    statistics: BackupStatistics,
    line_number: u64,
}

impl<R: Read> Importer<R> {
    /// Create a new importer with default options.
    pub fn new(reader: R) -> Self {
        Self::with_options(reader, ImportOptions::default())
    }

    /// Create a new importer with custom options.
    pub fn with_options(reader: R, options: ImportOptions) -> Self {
        Self {
            reader: BufReader::new(reader),
            metadata: None,
            options,
            statistics: BackupStatistics::default(),
            line_number: 0,
        }
    }

    /// Read and validate the backup metadata.
    ///
    /// This must be called before importing data.
    pub fn read_metadata(&mut self) -> BackupResult<&BackupMetadata> {
        if let Some(ref meta) = self.metadata {
            return Ok(meta);
        }

        let record =
            self.read_record()?.ok_or_else(|| BackupError::incomplete("empty backup file"))?;

        match record.record_type {
            BackupRecordType::Metadata => {
                if let RecordData::Metadata(meta) = record.data {
                    // Validate version
                    if meta.version > BACKUP_FORMAT_VERSION {
                        return Err(BackupError::UnsupportedVersion(meta.version));
                    }
                    Ok(self.metadata.insert(meta))
                } else {
                    Err(BackupError::malformed_record(
                        self.line_number,
                        "metadata record has wrong data type",
                    ))
                }
            }
            _ => Err(BackupError::malformed_record(
                self.line_number,
                "first record must be metadata",
            )),
        }
    }

    /// Get the backup metadata (after reading it).
    pub fn metadata(&self) -> Option<&BackupMetadata> {
        self.metadata.as_ref()
    }

    /// Get the current import statistics.
    pub fn statistics(&self) -> &BackupStatistics {
        &self.statistics
    }

    /// Read the next record from the backup.
    pub fn read_record(&mut self) -> BackupResult<Option<BackupRecord>> {
        let mut line = String::new();
        let bytes_read = self.reader.read_line(&mut line)?;

        if bytes_read == 0 {
            return Ok(None);
        }

        self.line_number += 1;
        let line = line.trim();

        if line.is_empty() {
            return self.read_record(); // Skip empty lines
        }

        let record: BackupRecord = serde_json::from_str(line).map_err(|e| {
            BackupError::Deserialization(format!("line {}: {}", self.line_number, e))
        })?;

        Ok(Some(record))
    }

    /// Import all records into the database.
    pub fn import_all(mut self, db: &Database) -> BackupResult<BackupStatistics> {
        // Read and validate metadata first
        self.read_metadata()?;

        if self.options.dry_run {
            return self.verify_all();
        }

        // Collect entity IDs for reference checking
        let mut entity_ids: HashSet<u64> = HashSet::new();

        // Process records in batches
        let batch_size = self.options.batch_size.unwrap_or(usize::MAX);
        let mut batch_records: Vec<BackupRecord> = Vec::with_capacity(batch_size.min(10_000));

        loop {
            // Read a batch of records
            batch_records.clear();

            for _ in 0..batch_size {
                match self.read_record()? {
                    Some(record) => {
                        if matches!(record.record_type, BackupRecordType::EndOfBackup) {
                            break;
                        }
                        batch_records.push(record);
                    }
                    None => break,
                }
            }

            if batch_records.is_empty() {
                break;
            }

            // Process this batch in a transaction
            let mut tx = db.begin()?;

            for record in &batch_records {
                match &record.record_type {
                    BackupRecordType::Entity => {
                        if let RecordData::Entity(entity_record) = &record.data {
                            entity_ids.insert(entity_record.id);
                            let entity = entity_record.to_entity();
                            tx.put_entity(&entity)?;
                            self.statistics.add_entity();
                        }
                    }
                    BackupRecordType::Edge => {
                        if let RecordData::Edge(edge_record) = &record.data {
                            // Verify references if enabled
                            if self.options.verify_references {
                                if !entity_ids.contains(&edge_record.source) {
                                    return Err(BackupError::MissingReference(format!(
                                        "edge {} references missing source entity {}",
                                        edge_record.id, edge_record.source
                                    )));
                                }
                                if !entity_ids.contains(&edge_record.target) {
                                    return Err(BackupError::MissingReference(format!(
                                        "edge {} references missing target entity {}",
                                        edge_record.id, edge_record.target
                                    )));
                                }
                            }

                            let edge = edge_record.to_edge();
                            tx.put_edge(&edge)?;
                            self.statistics.add_edge();
                        }
                    }
                    BackupRecordType::KeyValue => {
                        if let RecordData::KeyValue(kv) = &record.data {
                            if let Some(table) = &record.table {
                                self.import_key_value(&mut tx, table, kv)?;
                                self.statistics.add_metadata();
                            }
                        }
                    }
                    BackupRecordType::Metadata | BackupRecordType::EndOfBackup => {
                        // Already handled or marker
                    }
                }
            }

            tx.commit()?;
        }

        Ok(self.statistics)
    }

    /// Import a key-value record.
    fn import_key_value<T: manifoldb_storage::Transaction>(
        &self,
        tx: &mut crate::transaction::DatabaseTransaction<T>,
        table: &str,
        kv: &KeyValueRecord,
    ) -> BackupResult<()> {
        // Only import metadata table
        if table == tables::METADATA {
            tx.put_metadata(&kv.key, &kv.value)?;
        }
        Ok(())
    }

    /// Verify all records without importing (dry run).
    fn verify_all(mut self) -> BackupResult<BackupStatistics> {
        let mut entity_ids: HashSet<u64> = HashSet::new();

        while let Some(record) = self.read_record()? {
            match &record.record_type {
                BackupRecordType::Entity => {
                    if let RecordData::Entity(entity_record) = &record.data {
                        entity_ids.insert(entity_record.id);
                        self.statistics.add_entity();
                    }
                }
                BackupRecordType::Edge => {
                    if let RecordData::Edge(edge_record) = &record.data {
                        if self.options.verify_references {
                            if !entity_ids.contains(&edge_record.source) {
                                return Err(BackupError::MissingReference(format!(
                                    "edge {} references missing source entity {}",
                                    edge_record.id, edge_record.source
                                )));
                            }
                            if !entity_ids.contains(&edge_record.target) {
                                return Err(BackupError::MissingReference(format!(
                                    "edge {} references missing target entity {}",
                                    edge_record.id, edge_record.target
                                )));
                            }
                        }
                        self.statistics.add_edge();
                    }
                }
                BackupRecordType::KeyValue => {
                    self.statistics.add_metadata();
                }
                BackupRecordType::EndOfBackup => {
                    break;
                }
                BackupRecordType::Metadata => {
                    // Already processed
                }
            }
        }

        Ok(self.statistics)
    }
}

/// Import a backup into the database.
///
/// This is a convenience function that creates an importer and runs the import.
///
/// # Arguments
///
/// * `db` - The database to restore into
/// * `reader` - The input reader containing backup data
///
/// # Returns
///
/// Returns statistics about the import on success.
///
/// # Examples
///
/// ```ignore
/// use manifoldb::{Database, backup};
/// use std::fs::File;
/// use std::io::BufReader;
///
/// let db = Database::open("restored.manifold")?;
/// let file = File::open("backup.jsonl")?;
/// let reader = BufReader::new(file);
///
/// let stats = backup::import(&db, reader)?;
/// println!("Restored {} entities", stats.entity_count);
/// ```
pub fn import<R: Read>(db: &Database, reader: R) -> BackupResult<BackupStatistics> {
    let importer = Importer::new(reader);
    importer.import_all(db)
}

/// Verify a backup without importing.
///
/// This performs a dry run of the import process, validating the backup
/// format and checking referential integrity without modifying the database.
///
/// # Arguments
///
/// * `reader` - The input reader containing backup data
///
/// # Returns
///
/// Returns statistics about what would be imported on success.
pub fn verify<R: Read>(reader: R) -> BackupResult<BackupStatistics> {
    let importer = Importer::with_options(reader, ImportOptions::dry_run());
    importer.import_all(&Database::in_memory()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::export::export_full;
    use std::io::Cursor;

    #[test]
    fn test_import_options_default() {
        let opts = ImportOptions::default();
        assert!(opts.verify_references);
        assert!(!opts.skip_duplicates);
        assert!(opts.rebuild_indexes);
        assert!(!opts.dry_run);
    }

    #[test]
    fn test_import_options_fast() {
        let opts = ImportOptions::fast();
        assert!(!opts.verify_references);
        assert!(opts.skip_duplicates);
        assert!(!opts.rebuild_indexes);
    }

    #[test]
    fn test_roundtrip_empty_db() {
        let source_db = Database::in_memory().unwrap();

        // Export
        let mut buffer = Vec::new();
        let _stats = export_full(&source_db, &mut buffer).unwrap();

        // Import into new database
        let target_db = Database::in_memory().unwrap();
        let cursor = Cursor::new(buffer);
        let stats = import(&target_db, cursor).unwrap();

        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.edge_count, 0);
    }

    #[test]
    fn test_roundtrip_with_data() {
        let source_db = Database::in_memory().unwrap();

        // Add data to source
        {
            let mut tx = source_db.begin().unwrap();
            let e1 =
                tx.create_entity().unwrap().with_label("Person").with_property("name", "Alice");
            let e2 = tx.create_entity().unwrap().with_label("Person").with_property("name", "Bob");
            tx.put_entity(&e1).unwrap();
            tx.put_entity(&e2).unwrap();

            let edge =
                tx.create_edge(e1.id, e2.id, "KNOWS").unwrap().with_property("since", 2024i64);
            tx.put_edge(&edge).unwrap();
            tx.commit().unwrap();
        }

        // Export
        let mut buffer = Vec::new();
        let export_stats = export_full(&source_db, &mut buffer).unwrap();
        assert_eq!(export_stats.entity_count, 2);
        assert_eq!(export_stats.edge_count, 1);

        // Import into new database
        let target_db = Database::in_memory().unwrap();
        let cursor = Cursor::new(buffer);
        let import_stats = import(&target_db, cursor).unwrap();

        assert_eq!(import_stats.entity_count, 2);
        assert_eq!(import_stats.edge_count, 1);

        // Verify data was restored correctly
        let tx = target_db.begin_read().unwrap();
        let entities = tx.iter_entities(Some("Person")).unwrap();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_verify_backup() {
        let db = Database::in_memory().unwrap();

        // Add data
        {
            let mut tx = db.begin().unwrap();
            let e1 = tx.create_entity().unwrap().with_label("Test");
            let e2 = tx.create_entity().unwrap().with_label("Test");
            tx.put_entity(&e1).unwrap();
            tx.put_entity(&e2).unwrap();

            let edge = tx.create_edge(e1.id, e2.id, "LINKS").unwrap();
            tx.put_edge(&edge).unwrap();
            tx.commit().unwrap();
        }

        // Export
        let mut buffer = Vec::new();
        export_full(&db, &mut buffer).unwrap();

        // Verify (dry run)
        let cursor = Cursor::new(buffer);
        let stats = verify(cursor).unwrap();

        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[test]
    fn test_missing_reference_detection() {
        // Create a backup with an edge referencing non-existent entities
        let backup = r#"{"type":"metadata","data":{"version":1,"format":"json_lines","created_at":0,"sequence_number":0,"is_incremental":false,"previous_sequence":null,"statistics":{"entity_count":0,"edge_count":0,"metadata_count":0,"total_records":0,"uncompressed_size":0}}}
{"type":"edge","data":{"id":1,"source":999,"target":998,"edge_type":"BROKEN","properties":{}}}"#;

        let cursor = Cursor::new(backup.as_bytes());
        let result = verify(cursor);

        assert!(result.is_err());
        match result {
            Err(BackupError::MissingReference(_)) => (),
            _ => panic!("expected MissingReference error"),
        }
    }

    #[test]
    fn test_empty_backup_file() {
        let cursor = Cursor::new(b"");
        let result = verify(cursor);

        assert!(result.is_err());
        match result {
            Err(BackupError::Incomplete(msg)) => {
                assert!(msg.contains("empty"), "Expected 'empty' in message: {msg}");
            }
            other => panic!("expected Incomplete error, got: {other:?}"),
        }
    }

    #[test]
    fn test_truncated_json_line() {
        // Valid metadata followed by truncated JSON
        let backup = r#"{"type":"metadata","data":{"version":1,"format":"json_lines","created_at":0,"sequence_number":0,"is_incremental":false,"previous_sequence":null,"statistics":{"entity_count":0,"edge_count":0,"metadata_count":0,"total_records":0,"uncompressed_size":0}}}
{"type":"entity","data":{"id":1,"labels":["Test"],"properties":{"#;

        let cursor = Cursor::new(backup.as_bytes());
        let result = verify(cursor);

        assert!(result.is_err());
        match result {
            Err(BackupError::Deserialization(msg)) => {
                assert!(msg.contains("line 2"), "Expected line number in message: {msg}");
            }
            other => panic!("expected Deserialization error, got: {other:?}"),
        }
    }

    #[test]
    fn test_invalid_json_syntax() {
        // Invalid JSON on first line
        let backup = "not valid json at all";

        let cursor = Cursor::new(backup.as_bytes());
        let result = verify(cursor);

        assert!(result.is_err());
        match result {
            Err(BackupError::Deserialization(msg)) => {
                assert!(msg.contains("line 1"), "Expected line number in message: {msg}");
            }
            other => panic!("expected Deserialization error, got: {other:?}"),
        }
    }

    #[test]
    fn test_non_metadata_first_record() {
        // Entity record instead of metadata first
        let backup = r#"{"type":"entity","data":{"id":1,"labels":["Test"],"properties":{}}}"#;

        let cursor = Cursor::new(backup.as_bytes());
        let result = verify(cursor);

        assert!(result.is_err());
        match result {
            Err(BackupError::MalformedRecord { line, message }) => {
                assert_eq!(line, 1);
                assert!(
                    message.contains("first record must be metadata"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("expected MalformedRecord error, got: {other:?}"),
        }
    }

    #[test]
    fn test_unsupported_version() {
        // Future version that shouldn't be supported
        let backup = r#"{"type":"metadata","data":{"version":999,"format":"json_lines","created_at":0,"sequence_number":0,"is_incremental":false,"previous_sequence":null,"statistics":{"entity_count":0,"edge_count":0,"metadata_count":0,"total_records":0,"uncompressed_size":0}}}"#;

        let cursor = Cursor::new(backup.as_bytes());
        let result = verify(cursor);

        assert!(result.is_err());
        match result {
            Err(BackupError::UnsupportedVersion(version)) => {
                assert_eq!(version, 999);
            }
            other => panic!("expected UnsupportedVersion error, got: {other:?}"),
        }
    }

    #[test]
    fn test_corrupted_entity_record() {
        // Valid metadata but entity record with wrong data type
        let backup = r#"{"type":"metadata","data":{"version":1,"format":"json_lines","created_at":0,"sequence_number":0,"is_incremental":false,"previous_sequence":null,"statistics":{"entity_count":0,"edge_count":0,"metadata_count":0,"total_records":0,"uncompressed_size":0}}}
{"type":"entity","data":{"wrong_field":"bad_value"}}"#;

        let cursor = Cursor::new(backup.as_bytes());
        let result = verify(cursor);

        // This should fail during deserialization because the data doesn't match EntityRecord
        assert!(result.is_err());
    }
}
