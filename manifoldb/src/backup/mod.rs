//! Backup and restore utilities for ManifoldDB.
//!
//! This module provides functionality to export and import database contents
//! for backup, migration, and disaster recovery purposes.
//!
//! # Features
//!
//! - **Full backup**: Export all data to a portable JSON-lines format
//! - **Incremental backup**: Export only changes since a given sequence number
//! - **Restore**: Load backup data into a fresh or existing database
//! - **Verification**: Validate backup integrity before restore
//!
//! # Format
//!
//! Backups use a JSON-lines format with the following structure:
//! - Line 1: Backup metadata (version, timestamp, sequence, statistics)
//! - Line 2+: Data records (entities, edges, metadata, etc.)
//!
//! This format is human-readable, streamable, and easy to process with
//! standard Unix tools.
//!
//! # Examples
//!
//! ## Full Backup
//!
//! ```ignore
//! use manifoldb::{Database, backup};
//! use std::fs::File;
//! use std::io::BufWriter;
//!
//! let db = Database::open("mydb.manifold")?;
//! let file = File::create("backup.jsonl")?;
//! let writer = BufWriter::new(file);
//!
//! let stats = backup::export_full(&db, writer)?;
//! println!("Backed up {} entities and {} edges", stats.entity_count, stats.edge_count);
//! ```
//!
//! ## Restore
//!
//! ```ignore
//! use manifoldb::{Database, backup};
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! let db = Database::open("restored.manifold")?;
//! let file = File::open("backup.jsonl")?;
//! let reader = BufReader::new(file);
//!
//! let stats = backup::import(&db, reader)?;
//! println!("Restored {} entities and {} edges", stats.entity_count, stats.edge_count);
//! ```
//!
//! # Incremental Backups
//!
//! Incremental backups track changes using a sequence number stored in the
//! database metadata. Each write transaction increments this sequence number,
//! allowing subsequent backups to export only new or modified data.
//!
//! Note: True incremental backup requires tracking per-record modification times
//! or using change data capture. The current implementation provides a simplified
//! version that tracks the backup sequence number for record-keeping purposes.
//!
//! # Point-in-Time Recovery
//!
//! Full point-in-time recovery would require write-ahead log (WAL) access, which
//! is not currently exposed by the underlying redb storage engine. The backup
//! system provides consistent snapshots via read transactions, but cannot replay
//! individual operations for point-in-time recovery.
//!
//! For disaster recovery, we recommend:
//! - Regular full backups (e.g., daily)
//! - Incremental backups between full backups
//! - Storing backups in multiple locations

mod error;
mod export;
mod import;
mod types;

pub use error::{BackupError, BackupResult};
pub use export::{export_full, export_incremental, BackupWriter};
pub use import::{import, verify, ImportOptions, Importer};
pub use types::{
    BackupFormat, BackupMetadata, BackupRecord, BackupRecordType, BackupStatistics, RecordData,
};
