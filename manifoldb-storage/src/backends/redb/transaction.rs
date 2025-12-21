//! Redb transaction implementation.
//!
//! This module provides the `RedbTransaction` type which implements the
//! `Transaction` trait for both read-only and read-write transactions.

use std::ops::Bound;

use redb::{ReadTransaction, ReadableTable, WriteTransaction};

use crate::engine::{Cursor, CursorResult, KeyValue, StorageError, Transaction};

use super::tables::{decode_key, encode_key, table_end_key, table_start_key, DATA_TABLE};

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
        = RedbCursor
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
        // Collect all entries from this logical table
        let entries = self.collect_table_entries(table)?;
        Ok(RedbCursor::new(entries, None, None))
    }

    fn range(
        &self,
        table: &str,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
    ) -> Result<Self::Cursor<'_>, StorageError> {
        let entries = self.collect_table_entries(table)?;
        let start_owned = bound_to_owned(start);
        let end_owned = bound_to_owned(end);
        Ok(RedbCursor::new(entries, Some(start_owned), Some(end_owned)))
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
    /// Collect all entries from a logical table into a vector.
    fn collect_table_entries(&self, table: &str) -> Result<Vec<KeyValue>, StorageError> {
        let start = table_start_key(table);
        let end = table_end_key(table);

        match self {
            Self::Read(tx) => match tx.open_table(DATA_TABLE) {
                Ok(t) => {
                    let mut entries = Vec::new();
                    let range = t
                        .range(start.as_slice()..end.as_slice())
                        .map_err(|e| StorageError::Internal(e.to_string()))?;
                    for result in range {
                        let (k, v) = result.map_err(|e| StorageError::Internal(e.to_string()))?;
                        // Decode to get the original key (without table prefix)
                        if let Some((_, original_key)) = decode_key(k.value()) {
                            entries.push((original_key.to_vec(), v.value().to_vec()));
                        }
                    }
                    Ok(entries)
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    Err(StorageError::TableNotFound(table.to_string()))
                }
                Err(e) => Err(StorageError::Internal(e.to_string())),
            },
            Self::Write(tx) => match tx.open_table(DATA_TABLE) {
                Ok(t) => {
                    let mut entries = Vec::new();
                    let range = t
                        .range(start.as_slice()..end.as_slice())
                        .map_err(|e| StorageError::Internal(e.to_string()))?;
                    for result in range {
                        let (k, v) = result.map_err(|e| StorageError::Internal(e.to_string()))?;
                        // Decode to get the original key (without table prefix)
                        if let Some((_, original_key)) = decode_key(k.value()) {
                            entries.push((original_key.to_vec(), v.value().to_vec()));
                        }
                    }
                    Ok(entries)
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    Err(StorageError::TableNotFound(table.to_string()))
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

/// A cursor for iterating over key-value pairs in Redb.
///
/// This implementation stores a snapshot of the table entries in memory
/// to provide stable iteration without holding table references.
pub struct RedbCursor {
    /// All entries in the table (or range).
    entries: Vec<KeyValue>,
    /// Current position in the entries vector.
    /// None means "before first" position.
    position: Option<usize>,
    /// Optional start bound for range queries.
    start_bound: Option<Bound<Vec<u8>>>,
    /// Optional end bound for range queries.
    end_bound: Option<Bound<Vec<u8>>>,
}

impl RedbCursor {
    /// Create a new cursor with the given entries and optional bounds.
    pub const fn new(
        entries: Vec<KeyValue>,
        start_bound: Option<Bound<Vec<u8>>>,
        end_bound: Option<Bound<Vec<u8>>>,
    ) -> Self {
        Self { entries, position: None, start_bound, end_bound }
    }

    /// Check if a key is within the cursor's bounds.
    fn in_bounds(&self, key: &[u8]) -> bool {
        let after_start = match &self.start_bound {
            None | Some(Bound::Unbounded) => true,
            Some(Bound::Included(start)) => key >= start.as_slice(),
            Some(Bound::Excluded(start)) => key > start.as_slice(),
        };

        let before_end = match &self.end_bound {
            None | Some(Bound::Unbounded) => true,
            Some(Bound::Included(end)) => key <= end.as_slice(),
            Some(Bound::Excluded(end)) => key < end.as_slice(),
        };

        after_start && before_end
    }

    /// Find the first valid position (respecting bounds).
    fn find_first_valid(&self) -> Option<usize> {
        for (i, (key, _)) in self.entries.iter().enumerate() {
            if self.in_bounds(key) {
                return Some(i);
            }
        }
        None
    }

    /// Find the last valid position (respecting bounds).
    fn find_last_valid(&self) -> Option<usize> {
        for (i, (key, _)) in self.entries.iter().enumerate().rev() {
            if self.in_bounds(key) {
                return Some(i);
            }
        }
        None
    }
}

impl Cursor for RedbCursor {
    fn seek(&mut self, key: &[u8]) -> CursorResult {
        // Find the first entry with key >= target
        for (i, (k, v)) in self.entries.iter().enumerate() {
            if k.as_slice() >= key && self.in_bounds(k) {
                self.position = Some(i);
                return Ok(Some((k.clone(), v.clone())));
            }
        }
        self.position = None;
        Ok(None)
    }

    fn seek_first(&mut self) -> CursorResult {
        if let Some(i) = self.find_first_valid() {
            self.position = Some(i);
            let (k, v) = &self.entries[i];
            Ok(Some((k.clone(), v.clone())))
        } else {
            self.position = None;
            Ok(None)
        }
    }

    fn seek_last(&mut self) -> CursorResult {
        if let Some(i) = self.find_last_valid() {
            self.position = Some(i);
            let (k, v) = &self.entries[i];
            Ok(Some((k.clone(), v.clone())))
        } else {
            self.position = None;
            Ok(None)
        }
    }

    fn next(&mut self) -> CursorResult {
        let next_pos = match self.position {
            None => self.find_first_valid(),
            Some(current) => {
                // Find next valid position after current
                for i in (current + 1)..self.entries.len() {
                    if self.in_bounds(&self.entries[i].0) {
                        return {
                            self.position = Some(i);
                            let (k, v) = &self.entries[i];
                            Ok(Some((k.clone(), v.clone())))
                        };
                    }
                }
                None
            }
        };

        if let Some(i) = next_pos {
            self.position = Some(i);
            let (k, v) = &self.entries[i];
            Ok(Some((k.clone(), v.clone())))
        } else {
            self.position = None;
            Ok(None)
        }
    }

    fn prev(&mut self) -> CursorResult {
        match self.position {
            None => {
                // If no position, seek to last
                self.seek_last()
            }
            Some(0) => {
                self.position = None;
                Ok(None)
            }
            Some(current) => {
                // Find previous valid position
                for i in (0..current).rev() {
                    if self.in_bounds(&self.entries[i].0) {
                        self.position = Some(i);
                        let (k, v) = &self.entries[i];
                        return Ok(Some((k.clone(), v.clone())));
                    }
                }
                self.position = None;
                Ok(None)
            }
        }
    }

    fn current(&self) -> Option<(&[u8], &[u8])> {
        self.position.and_then(|i| {
            self.entries
                .get(i)
                .filter(|(k, _)| self.in_bounds(k))
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_empty() {
        let mut cursor = RedbCursor::new(vec![], None, None);
        assert!(cursor.seek_first().unwrap().is_none());
        assert!(cursor.seek_last().unwrap().is_none());
        assert!(cursor.next().unwrap().is_none());
        assert!(cursor.current().is_none());
    }

    #[test]
    fn test_cursor_single_entry() {
        let entries = vec![(b"key".to_vec(), b"value".to_vec())];
        let mut cursor = RedbCursor::new(entries, None, None);

        let first = cursor.seek_first().unwrap();
        assert_eq!(first, Some((b"key".to_vec(), b"value".to_vec())));
        assert_eq!(cursor.current(), Some((b"key".as_slice(), b"value".as_slice())));

        let next = cursor.next().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_cursor_multiple_entries() {
        let entries = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
        ];
        let mut cursor = RedbCursor::new(entries, None, None);

        // Forward iteration
        cursor.seek_first().unwrap();
        assert_eq!(cursor.current(), Some((b"a".as_slice(), b"1".as_slice())));

        cursor.next().unwrap();
        assert_eq!(cursor.current(), Some((b"b".as_slice(), b"2".as_slice())));

        cursor.next().unwrap();
        assert_eq!(cursor.current(), Some((b"c".as_slice(), b"3".as_slice())));

        // Backward
        cursor.prev().unwrap();
        assert_eq!(cursor.current(), Some((b"b".as_slice(), b"2".as_slice())));
    }

    #[test]
    fn test_cursor_seek() {
        let entries = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ];
        let mut cursor = RedbCursor::new(entries, None, None);

        // Seek to exact key
        let result = cursor.seek(b"c").unwrap();
        assert_eq!(result, Some((b"c".to_vec(), b"3".to_vec())));

        // Seek to non-existent key (should find next greater)
        let result = cursor.seek(b"b").unwrap();
        assert_eq!(result, Some((b"c".to_vec(), b"3".to_vec())));

        // Seek past all keys
        let result = cursor.seek(b"z").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cursor_with_bounds() {
        let entries = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ];
        let mut cursor = RedbCursor::new(
            entries,
            Some(Bound::Included(b"b".to_vec())),
            Some(Bound::Excluded(b"e".to_vec())),
        );

        // Should only iterate b, c, d
        let first = cursor.seek_first().unwrap();
        assert_eq!(first, Some((b"b".to_vec(), b"2".to_vec())));

        cursor.next().unwrap();
        assert_eq!(cursor.current(), Some((b"c".as_slice(), b"3".as_slice())));

        cursor.next().unwrap();
        assert_eq!(cursor.current(), Some((b"d".as_slice(), b"4".as_slice())));

        let past_end = cursor.next().unwrap();
        assert!(past_end.is_none());
    }
}
