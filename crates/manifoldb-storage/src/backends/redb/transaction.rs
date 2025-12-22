//! Redb transaction implementation.
//!
//! This module provides the `RedbTransaction` type which implements the
//! `Transaction` trait for both read-only and read-write transactions.
//!
//! # Memory-Efficient Cursors
//!
//! The cursor implementation uses batched streaming to avoid loading entire
//! tables into memory. Instead of materializing all entries upfront, it loads
//! entries in configurable batches (default 1000 entries), fetching the next
//! batch on demand as the cursor advances.

use std::ops::Bound;

use redb::{ReadTransaction, ReadableTable, WriteTransaction};

use crate::engine::{Cursor, CursorResult, KeyValue, StorageError, Transaction};

use super::tables::{decode_key, encode_key, table_end_key, table_start_key, DATA_TABLE};

/// Default batch size for cursor operations.
/// This limits memory usage while maintaining good performance.
const DEFAULT_BATCH_SIZE: usize = 1000;

/// A transaction for the Redb storage engine.
///
/// This type wraps both read-only and read-write Redb transactions,
/// providing a unified interface through the `Transaction` trait.
///
/// Note: We allow the `large_enum_variant` lint here because boxing the
/// `WriteTransaction` would add indirection overhead for every operation,
/// and transactions are typically short-lived.
#[allow(clippy::large_enum_variant)]
pub enum RedbTransaction {
    /// A read-only transaction.
    Read(ReadTransaction),
    /// A read-write transaction.
    Write(WriteTransaction),
}

impl RedbTransaction {
    /// Create a new read-only transaction.
    pub const fn new_read(tx: ReadTransaction) -> Self {
        Self::Read(tx)
    }

    /// Create a new read-write transaction.
    pub const fn new_write(tx: WriteTransaction) -> Self {
        Self::Write(tx)
    }
}

impl Transaction for RedbTransaction {
    type Cursor<'a>
        = RedbCursor<'a>
    where
        Self: 'a;

    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        let encoded_key = encode_key(table, key);

        match self {
            Self::Read(tx) => {
                match tx.open_table(DATA_TABLE) {
                    Ok(t) => match t.get(encoded_key.as_slice()) {
                        Ok(Some(value)) => Ok(Some(value.value().to_vec())),
                        Ok(None) => Ok(None),
                        Err(e) => Err(StorageError::Internal(e.to_string())),
                    },
                    Err(redb::TableError::TableDoesNotExist(_)) => {
                        // No data table means no data, which is not an error
                        Ok(None)
                    }
                    Err(e) => Err(StorageError::Internal(e.to_string())),
                }
            }
            Self::Write(tx) => {
                match tx.open_table(DATA_TABLE) {
                    Ok(t) => match t.get(encoded_key.as_slice()) {
                        Ok(Some(value)) => Ok(Some(value.value().to_vec())),
                        Ok(None) => Ok(None),
                        Err(e) => Err(StorageError::Internal(e.to_string())),
                    },
                    Err(redb::TableError::TableDoesNotExist(_)) => {
                        // No data table means no data, which is not an error
                        Ok(None)
                    }
                    Err(e) => Err(StorageError::Internal(e.to_string())),
                }
            }
        }
    }

    fn put(&mut self, table: &str, key: &[u8], value: &[u8]) -> Result<(), StorageError> {
        match self {
            Self::Read(_) => Err(StorageError::ReadOnly),
            Self::Write(tx) => {
                let encoded_key = encode_key(table, key);
                let mut t =
                    tx.open_table(DATA_TABLE).map_err(|e| StorageError::Internal(e.to_string()))?;
                t.insert(encoded_key.as_slice(), value)
                    .map_err(|e| StorageError::Internal(e.to_string()))?;
                Ok(())
            }
        }
    }

    fn delete(&mut self, table: &str, key: &[u8]) -> Result<bool, StorageError> {
        match self {
            Self::Read(_) => Err(StorageError::ReadOnly),
            Self::Write(tx) => {
                let encoded_key = encode_key(table, key);
                match tx.open_table(DATA_TABLE) {
                    Ok(mut t) => match t.remove(encoded_key.as_slice()) {
                        Ok(Some(_)) => Ok(true),
                        Ok(None) => Ok(false),
                        Err(e) => Err(StorageError::Internal(e.to_string())),
                    },
                    Err(redb::TableError::TableDoesNotExist(_)) => {
                        // Table doesn't exist, so key definitely doesn't exist
                        Ok(false)
                    }
                    Err(e) => Err(StorageError::Internal(e.to_string())),
                }
            }
        }
    }

    fn cursor(&self, table: &str) -> Result<Self::Cursor<'_>, StorageError> {
        RedbCursor::new(self, table.to_string(), None, None, DEFAULT_BATCH_SIZE)
    }

    fn range(
        &self,
        table: &str,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
    ) -> Result<Self::Cursor<'_>, StorageError> {
        let start_owned = bound_to_owned(start);
        let end_owned = bound_to_owned(end);
        RedbCursor::new(
            self,
            table.to_string(),
            Some(start_owned),
            Some(end_owned),
            DEFAULT_BATCH_SIZE,
        )
    }

    fn commit(self) -> Result<(), StorageError> {
        match self {
            Self::Read(_) => {
                // Read transactions don't need explicit commit
                Ok(())
            }
            Self::Write(tx) => tx.commit().map_err(|e| StorageError::Transaction(e.to_string())),
        }
    }

    fn rollback(self) -> Result<(), StorageError> {
        match self {
            Self::Read(_) => {
                // Read transactions just get dropped
                Ok(())
            }
            Self::Write(tx) => {
                // Ignore abort result - we're rolling back anyway
                drop(tx.abort());
                Ok(())
            }
        }
    }

    fn is_read_only(&self) -> bool {
        matches!(self, Self::Read(_))
    }
}

impl RedbTransaction {
    /// Fetch a batch of entries from the table, starting after the given key.
    ///
    /// This is the core method for batched streaming. It fetches up to `batch_size`
    /// entries starting from `after_key` (exclusive) or from the beginning if None.
    fn fetch_batch(
        &self,
        table: &str,
        after_key: Option<&[u8]>,
        user_start_bound: &Option<Bound<Vec<u8>>>,
        user_end_bound: &Option<Bound<Vec<u8>>>,
        batch_size: usize,
    ) -> Result<Vec<KeyValue>, StorageError> {
        // Compute the physical range for this table
        let table_start = table_start_key(table);
        let table_end = table_end_key(table);

        // Compute effective start based on after_key or user bounds
        let effective_start: Vec<u8> = if let Some(after) = after_key {
            // Start after the given key
            encode_key(table, after)
        } else {
            // Start from user's start bound or table start
            match user_start_bound {
                Some(Bound::Included(k)) => encode_key(table, k),
                Some(Bound::Excluded(k)) => encode_key(table, k),
                _ => table_start.clone(),
            }
        };

        // Determine if we should skip the first key (for after_key or Excluded bounds)
        let skip_first =
            after_key.is_some() || matches!(user_start_bound, Some(Bound::Excluded(_)));

        // Compute effective end based on user bounds
        let effective_end: Vec<u8> = match user_end_bound {
            Some(Bound::Included(k)) => {
                // We need to include k, so use encode_key(table, k) + 1 byte
                let mut end = encode_key(table, k);
                end.push(0xFF);
                end
            }
            Some(Bound::Excluded(k)) => encode_key(table, k),
            _ => table_end,
        };

        // Fetch entries
        let mut entries = Vec::with_capacity(batch_size.min(1024));

        match self {
            Self::Read(tx) => match tx.open_table(DATA_TABLE) {
                Ok(t) => {
                    let range = t
                        .range(effective_start.as_slice()..effective_end.as_slice())
                        .map_err(|e| StorageError::Internal(e.to_string()))?;

                    let mut skipped_first = !skip_first;
                    for result in range {
                        if entries.len() >= batch_size {
                            break;
                        }

                        let (k, v) = result.map_err(|e| StorageError::Internal(e.to_string()))?;
                        if let Some((_, original_key)) = decode_key(k.value()) {
                            // Skip the first entry if needed (for after_key continuation)
                            if !skipped_first {
                                skipped_first = true;
                                continue;
                            }

                            // Check user end bound for Included case
                            if let Some(Bound::Included(end_key)) = user_end_bound {
                                if original_key > end_key.as_slice() {
                                    break;
                                }
                            }

                            entries.push((original_key.to_vec(), v.value().to_vec()));
                        }
                    }
                    Ok(entries)
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    // Table doesn't exist yet, return empty result (not an error)
                    Ok(Vec::new())
                }
                Err(e) => Err(StorageError::Internal(e.to_string())),
            },
            Self::Write(tx) => match tx.open_table(DATA_TABLE) {
                Ok(t) => {
                    let range = t
                        .range(effective_start.as_slice()..effective_end.as_slice())
                        .map_err(|e| StorageError::Internal(e.to_string()))?;

                    let mut skipped_first = !skip_first;
                    for result in range {
                        if entries.len() >= batch_size {
                            break;
                        }

                        let (k, v) = result.map_err(|e| StorageError::Internal(e.to_string()))?;
                        if let Some((_, original_key)) = decode_key(k.value()) {
                            if !skipped_first {
                                skipped_first = true;
                                continue;
                            }

                            if let Some(Bound::Included(end_key)) = user_end_bound {
                                if original_key > end_key.as_slice() {
                                    break;
                                }
                            }

                            entries.push((original_key.to_vec(), v.value().to_vec()));
                        }
                    }
                    Ok(entries)
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    // Table doesn't exist yet, return empty result (not an error)
                    Ok(Vec::new())
                }
                Err(e) => Err(StorageError::Internal(e.to_string())),
            },
        }
    }

    /// Fetch a batch in reverse, ending before the given key.
    ///
    /// Returns entries in ascending key order (even though fetched in reverse).
    fn fetch_batch_reverse(
        &self,
        table: &str,
        before_key: Option<&[u8]>,
        user_start_bound: &Option<Bound<Vec<u8>>>,
        user_end_bound: &Option<Bound<Vec<u8>>>,
        batch_size: usize,
    ) -> Result<Vec<KeyValue>, StorageError> {
        let table_start = table_start_key(table);
        let table_end = table_end_key(table);

        // Compute effective end for reverse scan
        // When before_key is provided, we use an exclusive range ending at before_key
        // When before_key is None, we use the user's end bound
        let effective_end: Vec<u8> = if let Some(before) = before_key {
            // Range [start..before_key) already excludes before_key
            encode_key(table, before)
        } else {
            match user_end_bound {
                Some(Bound::Included(k)) => {
                    // Need to include k, so extend past it
                    let mut end = encode_key(table, k);
                    end.push(0xFF);
                    end
                }
                Some(Bound::Excluded(k)) => encode_key(table, k),
                _ => table_end,
            }
        };

        // Compute effective start
        let effective_start: Vec<u8> = match user_start_bound {
            Some(Bound::Included(k)) => encode_key(table, k),
            Some(Bound::Excluded(k)) => {
                let mut start = encode_key(table, k);
                start.push(0x00);
                start
            }
            _ => table_start,
        };

        let mut entries = Vec::with_capacity(batch_size.min(1024));

        match self {
            Self::Read(tx) => match tx.open_table(DATA_TABLE) {
                Ok(t) => {
                    let range = t
                        .range(effective_start.as_slice()..effective_end.as_slice())
                        .map_err(|e| StorageError::Internal(e.to_string()))?;

                    // Collect in reverse order by using rev()
                    for result in range.rev() {
                        if entries.len() >= batch_size {
                            break;
                        }

                        let (k, v) = result.map_err(|e| StorageError::Internal(e.to_string()))?;
                        if let Some((_, original_key)) = decode_key(k.value()) {
                            // Check start bound for Excluded case
                            if let Some(Bound::Excluded(start_key)) = user_start_bound {
                                if original_key <= start_key.as_slice() {
                                    break;
                                }
                            }

                            entries.push((original_key.to_vec(), v.value().to_vec()));
                        }
                    }
                    // Reverse to get ascending order within the batch
                    entries.reverse();
                    Ok(entries)
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    // Table doesn't exist yet, return empty result (not an error)
                    Ok(Vec::new())
                }
                Err(e) => Err(StorageError::Internal(e.to_string())),
            },
            Self::Write(tx) => match tx.open_table(DATA_TABLE) {
                Ok(t) => {
                    let range = t
                        .range(effective_start.as_slice()..effective_end.as_slice())
                        .map_err(|e| StorageError::Internal(e.to_string()))?;

                    for result in range.rev() {
                        if entries.len() >= batch_size {
                            break;
                        }

                        let (k, v) = result.map_err(|e| StorageError::Internal(e.to_string()))?;
                        if let Some((_, original_key)) = decode_key(k.value()) {
                            if let Some(Bound::Excluded(start_key)) = user_start_bound {
                                if original_key <= start_key.as_slice() {
                                    break;
                                }
                            }

                            entries.push((original_key.to_vec(), v.value().to_vec()));
                        }
                    }
                    entries.reverse();
                    Ok(entries)
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    // Table doesn't exist yet, return empty result (not an error)
                    Ok(Vec::new())
                }
                Err(e) => Err(StorageError::Internal(e.to_string())),
            },
        }
    }
}

/// Convert a `Bound<&[u8]>` to `Bound<Vec<u8>>`.
fn bound_to_owned(bound: Bound<&[u8]>) -> Bound<Vec<u8>> {
    match bound {
        Bound::Included(b) => Bound::Included(b.to_vec()),
        Bound::Excluded(b) => Bound::Excluded(b.to_vec()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// A memory-efficient cursor for iterating over key-value pairs in Redb.
///
/// This implementation uses batched streaming to avoid loading entire tables
/// into memory. Instead of materializing all entries upfront, it fetches
/// entries in batches and loads more data on demand as the cursor advances.
///
/// # Memory Guarantees
///
/// At any time, the cursor holds at most `batch_size` entries in memory,
/// plus the current entry (if any). This means a table with 1M entries
/// will use approximately the same memory as a table with 1K entries.
pub struct RedbCursor<'a> {
    /// Reference to the transaction for fetching additional batches.
    tx: &'a RedbTransaction,
    /// The logical table name.
    table: String,
    /// Current batch of entries.
    batch: Vec<KeyValue>,
    /// Position within the current batch.
    batch_position: Option<usize>,
    /// User's start bound for range queries.
    start_bound: Option<Bound<Vec<u8>>>,
    /// User's end bound for range queries.
    end_bound: Option<Bound<Vec<u8>>>,
    /// Maximum entries per batch.
    batch_size: usize,
    /// Whether there are more entries after the current batch.
    has_more_forward: bool,
    /// Whether there are more entries before the current batch.
    has_more_backward: bool,
    /// Cached current entry for the `current()` method.
    /// This is separate from the batch to handle edge cases.
    current_entry: Option<KeyValue>,
}

impl<'a> RedbCursor<'a> {
    /// Create a new streaming cursor.
    ///
    /// The cursor starts in an unpositioned state. Call `seek_first()`, `seek_last()`,
    /// or `seek()` to position the cursor before iterating.
    pub fn new(
        tx: &'a RedbTransaction,
        table: String,
        start_bound: Option<Bound<Vec<u8>>>,
        end_bound: Option<Bound<Vec<u8>>>,
        batch_size: usize,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            tx,
            table,
            batch: Vec::new(),
            batch_position: None,
            start_bound,
            end_bound,
            batch_size,
            has_more_forward: true,
            has_more_backward: true,
            current_entry: None,
        })
    }

    /// Load the first batch of entries.
    fn load_first_batch(&mut self) -> Result<(), StorageError> {
        self.batch = self.tx.fetch_batch(
            &self.table,
            None,
            &self.start_bound,
            &self.end_bound,
            self.batch_size,
        )?;
        self.has_more_forward = self.batch.len() >= self.batch_size;
        self.has_more_backward = false; // We're at the start
        Ok(())
    }

    /// Load the last batch of entries.
    fn load_last_batch(&mut self) -> Result<(), StorageError> {
        self.batch = self.tx.fetch_batch_reverse(
            &self.table,
            None,
            &self.start_bound,
            &self.end_bound,
            self.batch_size,
        )?;
        self.has_more_backward = self.batch.len() >= self.batch_size;
        self.has_more_forward = false; // We're at the end
        Ok(())
    }

    /// Load the next batch, continuing from the last key in the current batch.
    fn load_next_batch(&mut self) -> Result<bool, StorageError> {
        if !self.has_more_forward {
            return Ok(false);
        }

        let after_key = self.batch.last().map(|(k, _)| k.as_slice());

        let new_batch = self.tx.fetch_batch(
            &self.table,
            after_key,
            &self.start_bound,
            &self.end_bound,
            self.batch_size,
        )?;

        if new_batch.is_empty() {
            self.has_more_forward = false;
            return Ok(false);
        }

        self.has_more_forward = new_batch.len() >= self.batch_size;
        self.has_more_backward = true;
        self.batch = new_batch;
        self.batch_position = Some(0);

        Ok(true)
    }

    /// Load the previous batch, ending before the first key in the current batch.
    fn load_prev_batch(&mut self) -> Result<bool, StorageError> {
        if !self.has_more_backward {
            return Ok(false);
        }

        let before_key = self.batch.first().map(|(k, _)| k.as_slice());

        let new_batch = self.tx.fetch_batch_reverse(
            &self.table,
            before_key,
            &self.start_bound,
            &self.end_bound,
            self.batch_size,
        )?;

        if new_batch.is_empty() {
            self.has_more_backward = false;
            return Ok(false);
        }

        self.has_more_backward = new_batch.len() >= self.batch_size;
        self.has_more_forward = true;
        self.batch = new_batch;
        self.batch_position = Some(self.batch.len() - 1);

        Ok(true)
    }

    /// Load a batch starting at or after the given key for seek operations.
    fn load_batch_at_key(&mut self, key: &[u8]) -> Result<(), StorageError> {
        // Create a temporary start bound that's at least the seek key
        let seek_start = match &self.start_bound {
            Some(Bound::Included(start)) if start.as_slice() > key => {
                Some(Bound::Included(start.clone()))
            }
            Some(Bound::Excluded(start)) if start.as_slice() >= key => {
                Some(Bound::Excluded(start.clone()))
            }
            _ => Some(Bound::Included(key.to_vec())),
        };

        self.batch = self.tx.fetch_batch(
            &self.table,
            None,
            &seek_start,
            &self.end_bound,
            self.batch_size,
        )?;

        self.has_more_forward = self.batch.len() >= self.batch_size;
        // There might be entries before the seek key
        self.has_more_backward = key > self.start_bound_key().unwrap_or(&[]);

        Ok(())
    }

    /// Get the start bound key, if any.
    fn start_bound_key(&self) -> Option<&[u8]> {
        match &self.start_bound {
            Some(Bound::Included(k) | Bound::Excluded(k)) => Some(k.as_slice()),
            _ => None,
        }
    }

    /// Update the current entry cache from the batch.
    fn update_current(&mut self) {
        self.current_entry = self.batch_position.and_then(|pos| self.batch.get(pos).cloned());
    }
}

impl Cursor for RedbCursor<'_> {
    fn seek(&mut self, key: &[u8]) -> CursorResult {
        self.load_batch_at_key(key)?;

        // Use binary search to find the first key >= target
        let pos = self.batch.partition_point(|(k, _)| k.as_slice() < key);

        if pos < self.batch.len() {
            self.batch_position = Some(pos);
            self.update_current();
            Ok(self.current_entry.clone())
        } else if self.has_more_forward {
            // The key might be in the next batch
            if self.load_next_batch()? {
                self.batch_position = Some(0);
                self.update_current();
                Ok(self.current_entry.clone())
            } else {
                self.batch_position = None;
                self.current_entry = None;
                Ok(None)
            }
        } else {
            self.batch_position = None;
            self.current_entry = None;
            Ok(None)
        }
    }

    fn seek_first(&mut self) -> CursorResult {
        self.load_first_batch()?;

        if self.batch.is_empty() {
            self.batch_position = None;
            self.current_entry = None;
            return Ok(None);
        }

        self.batch_position = Some(0);
        self.update_current();
        Ok(self.current_entry.clone())
    }

    fn seek_last(&mut self) -> CursorResult {
        self.load_last_batch()?;

        if self.batch.is_empty() {
            self.batch_position = None;
            self.current_entry = None;
            return Ok(None);
        }

        self.batch_position = Some(self.batch.len() - 1);
        self.update_current();
        Ok(self.current_entry.clone())
    }

    fn next(&mut self) -> CursorResult {
        match self.batch_position {
            None => {
                // Not positioned, start from first
                self.seek_first()
            }
            Some(pos) => {
                let next_pos = pos + 1;
                if next_pos < self.batch.len() {
                    // Move within current batch
                    self.batch_position = Some(next_pos);
                    self.update_current();
                    Ok(self.current_entry.clone())
                } else if self.load_next_batch()? {
                    // Moved to next batch
                    self.update_current();
                    Ok(self.current_entry.clone())
                } else {
                    // No more entries
                    self.batch_position = None;
                    self.current_entry = None;
                    Ok(None)
                }
            }
        }
    }

    fn prev(&mut self) -> CursorResult {
        match self.batch_position {
            None => {
                // Not positioned, start from last
                self.seek_last()
            }
            Some(0) => {
                // At beginning of batch, try to load previous
                if self.load_prev_batch()? {
                    self.update_current();
                    Ok(self.current_entry.clone())
                } else {
                    self.batch_position = None;
                    self.current_entry = None;
                    Ok(None)
                }
            }
            Some(pos) => {
                // Move within current batch
                self.batch_position = Some(pos - 1);
                self.update_current();
                Ok(self.current_entry.clone())
            }
        }
    }

    fn current(&self) -> Option<(&[u8], &[u8])> {
        self.current_entry.as_ref().map(|(k, v)| (k.as_slice(), v.as_slice()))
    }
}

// Note: Cursor tests have been moved to the integration tests in tests/redb_cursor.rs
// because the streaming cursor requires a real transaction context.
