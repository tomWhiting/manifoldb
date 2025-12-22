//! Backup export functionality.
//!
//! This module provides functions to export database contents to a portable
//! JSON-lines format for backup and migration purposes.

use std::io::Write;
use std::ops::Bound;

use manifoldb_storage::{Cursor, Transaction};

use super::error::{BackupError, BackupResult};
use super::types::{BackupMetadata, BackupRecord, BackupStatistics, EdgeRecord, EntityRecord};
use crate::Database;

/// Well-known table names that should be backed up.
mod tables {
    pub const NODES: &str = "nodes";
    pub const EDGES: &str = "edges";
    pub const METADATA: &str = "metadata";
}

/// Metadata keys used by the backup system.
mod metadata_keys {
    /// The sequence number for tracking changes.
    pub const BACKUP_SEQUENCE: &[u8] = b"_backup_sequence";
}

/// A backup writer that writes records to an output stream.
///
/// This struct provides a streaming interface for writing backup records,
/// allowing large databases to be backed up without loading everything into memory.
pub struct BackupWriter<W: Write> {
    writer: W,
    statistics: BackupStatistics,
    records_written: u64,
}

impl<W: Write> BackupWriter<W> {
    /// Create a new backup writer.
    pub fn new(writer: W) -> Self {
        Self { writer, statistics: BackupStatistics::default(), records_written: 0 }
    }

    /// Write the backup metadata header.
    ///
    /// This must be the first record written.
    pub fn write_metadata(&mut self, metadata: &BackupMetadata) -> BackupResult<()> {
        let record = BackupRecord::metadata(metadata.clone());
        self.write_record(&record)?;
        Ok(())
    }

    /// Write an entity record.
    pub fn write_entity(&mut self, entity: &manifoldb_core::Entity) -> BackupResult<()> {
        let entity_record = EntityRecord::from_entity(entity);
        let record = BackupRecord::entity(entity_record);
        self.write_record(&record)?;
        self.statistics.add_entity();
        Ok(())
    }

    /// Write an edge record.
    pub fn write_edge(&mut self, edge: &manifoldb_core::Edge) -> BackupResult<()> {
        let edge_record = EdgeRecord::from_edge(edge);
        let record = BackupRecord::edge(edge_record);
        self.write_record(&record)?;
        self.statistics.add_edge();
        Ok(())
    }

    /// Write a raw key-value record.
    pub fn write_key_value(
        &mut self,
        table: &str,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> BackupResult<()> {
        let record = BackupRecord::key_value(table.to_string(), key, value);
        self.write_record(&record)?;
        self.statistics.add_metadata();
        Ok(())
    }

    /// Write the end-of-backup marker.
    pub fn finish(mut self) -> BackupResult<BackupStatistics> {
        let record = BackupRecord::end_of_backup(self.statistics.clone());
        self.write_record(&record)?;
        self.writer.flush()?;
        Ok(self.statistics)
    }

    /// Get the current statistics.
    pub fn statistics(&self) -> &BackupStatistics {
        &self.statistics
    }

    /// Get the number of records written.
    pub fn records_written(&self) -> u64 {
        self.records_written
    }

    /// Write a single record as a JSON line.
    fn write_record(&mut self, record: &BackupRecord) -> BackupResult<()> {
        let json = serde_json::to_string(record).map_err(BackupError::serialization)?;
        self.statistics.add_size(json.len() as u64 + 1); // +1 for newline
        writeln!(self.writer, "{}", json)?;
        self.records_written += 1;
        Ok(())
    }
}

/// Export a full backup of the database.
///
/// This function exports all entities, edges, and metadata to the provided writer
/// in JSON-lines format. The backup is taken within a read transaction to ensure
/// consistency.
///
/// # Arguments
///
/// * `db` - The database to back up
/// * `writer` - The output writer (e.g., a file or buffer)
///
/// # Returns
///
/// Returns statistics about the backup on success.
///
/// # Examples
///
/// ```ignore
/// use manifoldb::{Database, backup};
/// use std::fs::File;
/// use std::io::BufWriter;
///
/// let db = Database::open("mydb.manifold")?;
/// let file = File::create("backup.jsonl")?;
/// let writer = BufWriter::new(file);
///
/// let stats = backup::export_full(&db, writer)?;
/// println!("Backed up {} entities", stats.entity_count);
/// ```
pub fn export_full<W: Write>(db: &Database, writer: W) -> BackupResult<BackupStatistics> {
    let tx = db.begin_read()?;
    let storage = tx.storage_ref()?;

    // Get the current sequence number
    let sequence_number = get_sequence_number(storage)?;

    // Create metadata
    let metadata = BackupMetadata::new_full(sequence_number);

    // Create the backup writer and write header
    let mut backup_writer = BackupWriter::new(writer);
    backup_writer.write_metadata(&metadata)?;

    // Export entities
    export_entities(storage, &mut backup_writer)?;

    // Export edges
    export_edges(storage, &mut backup_writer)?;

    // Export metadata (except index-related entries)
    export_metadata(storage, &mut backup_writer)?;

    // Finish and return statistics
    backup_writer.finish()
}

/// Export an incremental backup since the given sequence number.
///
/// **Note**: True incremental backup requires per-record modification tracking
/// which is not currently implemented. This function exports all data but
/// records the sequence numbers for tracking purposes.
///
/// For a production-grade incremental backup, you would need to:
/// 1. Track modification timestamps on each record
/// 2. Use change data capture (CDC) mechanisms
/// 3. Or maintain a write-ahead log (WAL) that can be replayed
///
/// # Arguments
///
/// * `db` - The database to back up
/// * `writer` - The output writer
/// * `since_sequence` - Export changes since this sequence number
///
/// # Returns
///
/// Returns statistics about the backup on success.
pub fn export_incremental<W: Write>(
    db: &Database,
    writer: W,
    since_sequence: u64,
) -> BackupResult<BackupStatistics> {
    let tx = db.begin_read()?;
    let storage = tx.storage_ref()?;

    // Get the current sequence number
    let current_sequence = get_sequence_number(storage)?;

    // Create metadata
    let metadata = BackupMetadata::new_incremental(current_sequence, since_sequence);

    // Create the backup writer and write header
    let mut backup_writer = BackupWriter::new(writer);
    backup_writer.write_metadata(&metadata)?;

    // For now, export everything (true incremental requires modification tracking)
    // In a real implementation, you would filter records by their modification time
    export_entities(storage, &mut backup_writer)?;
    export_edges(storage, &mut backup_writer)?;
    export_metadata(storage, &mut backup_writer)?;

    // Finish and return statistics
    backup_writer.finish()
}

/// Get the current backup sequence number from metadata.
fn get_sequence_number<T: Transaction>(storage: &T) -> BackupResult<u64> {
    match storage.get(tables::METADATA, metadata_keys::BACKUP_SEQUENCE) {
        Ok(Some(bytes)) if bytes.len() == 8 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                BackupError::InvalidFormat("invalid sequence number format".to_string())
            })?;
            Ok(u64::from_be_bytes(arr))
        }
        Ok(_) => Ok(0), // No sequence number set yet
        Err(manifoldb_storage::StorageError::TableNotFound(_)) => Ok(0),
        Err(e) => Err(BackupError::Storage(e)),
    }
}

/// Export all entities from the database.
fn export_entities<T: Transaction, W: Write>(
    storage: &T,
    writer: &mut BackupWriter<W>,
) -> BackupResult<()> {
    let cursor_result = storage.range(tables::NODES, Bound::Unbounded, Bound::Unbounded);

    let mut cursor = match cursor_result {
        Ok(c) => c,
        Err(manifoldb_storage::StorageError::TableNotFound(_)) => return Ok(()),
        Err(e) => return Err(BackupError::Storage(e)),
    };

    while let Some((_key, value)) = cursor.next()? {
        let (entity, _): (manifoldb_core::Entity, _) =
            bincode::serde::decode_from_slice(&value, bincode::config::standard())
                .map_err(|e| BackupError::Deserialization(e.to_string()))?;
        writer.write_entity(&entity)?;
    }

    Ok(())
}

/// Export all edges from the database.
fn export_edges<T: Transaction, W: Write>(
    storage: &T,
    writer: &mut BackupWriter<W>,
) -> BackupResult<()> {
    let cursor_result = storage.range(tables::EDGES, Bound::Unbounded, Bound::Unbounded);

    let mut cursor = match cursor_result {
        Ok(c) => c,
        Err(manifoldb_storage::StorageError::TableNotFound(_)) => return Ok(()),
        Err(e) => return Err(BackupError::Storage(e)),
    };

    while let Some((_key, value)) = cursor.next()? {
        let (edge, _): (manifoldb_core::Edge, _) =
            bincode::serde::decode_from_slice(&value, bincode::config::standard())
                .map_err(|e| BackupError::Deserialization(e.to_string()))?;
        writer.write_edge(&edge)?;
    }

    Ok(())
}

/// Export metadata (non-index) from the database.
fn export_metadata<T: Transaction, W: Write>(
    storage: &T,
    writer: &mut BackupWriter<W>,
) -> BackupResult<()> {
    let cursor_result = storage.range(tables::METADATA, Bound::Unbounded, Bound::Unbounded);

    let mut cursor = match cursor_result {
        Ok(c) => c,
        Err(manifoldb_storage::StorageError::TableNotFound(_)) => return Ok(()),
        Err(e) => return Err(BackupError::Storage(e)),
    };

    while let Some((key, value)) = cursor.next()? {
        // Skip internal backup sequence key (it will be regenerated)
        if key.as_slice() == metadata_keys::BACKUP_SEQUENCE {
            continue;
        }

        writer.write_key_value(tables::METADATA, key, value)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_backup_writer_basic() {
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut writer = BackupWriter::new(cursor);

        let metadata = BackupMetadata::new_full(0);
        writer.write_metadata(&metadata).unwrap();

        let entity = manifoldb_core::Entity::new(manifoldb_core::EntityId::new(1))
            .with_label("Test")
            .with_property("name", "Test Entity");
        writer.write_entity(&entity).unwrap();

        let stats = writer.finish().unwrap();
        assert_eq!(stats.entity_count, 1);
        assert_eq!(stats.total_records, 1);
    }

    #[test]
    fn test_export_full_empty_db() {
        let db = Database::in_memory().unwrap();
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);

        let stats = export_full(&db, cursor).unwrap();
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.edge_count, 0);
    }

    #[test]
    fn test_export_full_with_data() {
        let db = Database::in_memory().unwrap();

        // Add some data
        {
            let mut tx = db.begin().unwrap();
            let entity1 =
                tx.create_entity().unwrap().with_label("Person").with_property("name", "Alice");
            let entity2 =
                tx.create_entity().unwrap().with_label("Person").with_property("name", "Bob");
            tx.put_entity(&entity1).unwrap();
            tx.put_entity(&entity2).unwrap();

            let edge = tx.create_edge(entity1.id, entity2.id, "KNOWS").unwrap();
            tx.put_edge(&edge).unwrap();
            tx.commit().unwrap();
        }

        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);

        let stats = export_full(&db, cursor).unwrap();
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.edge_count, 1);
    }
}
