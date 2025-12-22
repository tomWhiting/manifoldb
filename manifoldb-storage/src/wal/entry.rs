//! WAL entry types and serialization

use super::{Lsn, TxnId};
use serde::{Deserialize, Serialize};

/// Operation type for WAL entries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// Put a key-value pair
    Put,
    /// Delete a key
    Delete,
    /// Begin a transaction
    BeginTxn,
    /// Commit a transaction (makes all operations in txn durable)
    CommitTxn,
    /// Abort/rollback a transaction
    AbortTxn,
    /// Checkpoint marker - all entries before this LSN have been flushed to main storage
    Checkpoint,
}

/// A single entry in the write-ahead log
///
/// Each entry represents one operation that was or will be applied to the storage.
/// Entries are written atomically and include a checksum for corruption detection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalEntry {
    /// Log Sequence Number - monotonically increasing identifier
    pub lsn: Lsn,

    /// Transaction ID this operation belongs to (0 for standalone operations)
    pub txn_id: TxnId,

    /// Type of operation
    pub operation: Operation,

    /// Logical table name (None for transaction markers)
    pub table: Option<String>,

    /// Key being operated on (None for transaction markers)
    pub key: Option<Vec<u8>>,

    /// Value for Put operations (None for Delete and markers)
    pub value: Option<Vec<u8>>,

    /// Timestamp when this entry was created (Unix epoch millis)
    pub timestamp: u64,
}

impl WalEntry {
    /// Create a Put entry without transaction context
    pub fn put(lsn: Lsn, table: impl Into<String>, key: &[u8], value: &[u8]) -> Self {
        Self {
            lsn,
            txn_id: 0,
            operation: Operation::Put,
            table: Some(table.into()),
            key: Some(key.to_vec()),
            value: Some(value.to_vec()),
            timestamp: current_timestamp(),
        }
    }

    /// Create a Delete entry without transaction context
    pub fn delete(lsn: Lsn, table: impl Into<String>, key: &[u8]) -> Self {
        Self {
            lsn,
            txn_id: 0,
            operation: Operation::Delete,
            table: Some(table.into()),
            key: Some(key.to_vec()),
            value: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create a Put entry within a transaction
    pub fn put_in_txn(
        lsn: Lsn,
        txn_id: TxnId,
        table: impl Into<String>,
        key: &[u8],
        value: &[u8],
    ) -> Self {
        Self {
            lsn,
            txn_id,
            operation: Operation::Put,
            table: Some(table.into()),
            key: Some(key.to_vec()),
            value: Some(value.to_vec()),
            timestamp: current_timestamp(),
        }
    }

    /// Create a Delete entry within a transaction
    pub fn delete_in_txn(lsn: Lsn, txn_id: TxnId, table: impl Into<String>, key: &[u8]) -> Self {
        Self {
            lsn,
            txn_id,
            operation: Operation::Delete,
            table: Some(table.into()),
            key: Some(key.to_vec()),
            value: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create a BeginTxn marker
    pub fn begin_txn(lsn: Lsn, txn_id: TxnId) -> Self {
        Self {
            lsn,
            txn_id,
            operation: Operation::BeginTxn,
            table: None,
            key: None,
            value: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create a CommitTxn marker
    pub fn commit_txn(lsn: Lsn, txn_id: TxnId) -> Self {
        Self {
            lsn,
            txn_id,
            operation: Operation::CommitTxn,
            table: None,
            key: None,
            value: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create an AbortTxn marker
    pub fn abort_txn(lsn: Lsn, txn_id: TxnId) -> Self {
        Self {
            lsn,
            txn_id,
            operation: Operation::AbortTxn,
            table: None,
            key: None,
            value: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create a Checkpoint marker
    pub fn checkpoint(lsn: Lsn, checkpoint_lsn: Lsn) -> Self {
        // Store the checkpoint LSN in the key field as bytes
        Self {
            lsn,
            txn_id: 0,
            operation: Operation::Checkpoint,
            table: None,
            key: Some(checkpoint_lsn.to_le_bytes().to_vec()),
            value: None,
            timestamp: current_timestamp(),
        }
    }

    /// Returns true if this is a data operation (Put or Delete)
    pub const fn is_data_operation(&self) -> bool {
        matches!(self.operation, Operation::Put | Operation::Delete)
    }

    /// Returns true if this is a transaction boundary marker
    pub const fn is_txn_marker(&self) -> bool {
        matches!(self.operation, Operation::BeginTxn | Operation::CommitTxn | Operation::AbortTxn)
    }

    /// Returns true if this is a checkpoint marker
    pub const fn is_checkpoint(&self) -> bool {
        matches!(self.operation, Operation::Checkpoint)
    }

    /// Get the checkpoint LSN if this is a checkpoint entry
    pub fn checkpoint_lsn(&self) -> Option<Lsn> {
        if self.operation == Operation::Checkpoint {
            self.key.as_ref().and_then(|k| {
                if k.len() == 8 {
                    Some(Lsn::from_le_bytes(k[..8].try_into().ok()?))
                } else {
                    None
                }
            })
        } else {
            None
        }
    }
}

/// Get current timestamp in milliseconds since Unix epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_entry() {
        let entry = WalEntry::put(1, "nodes", b"key", b"value");
        assert_eq!(entry.lsn, 1);
        assert_eq!(entry.txn_id, 0);
        assert!(matches!(entry.operation, Operation::Put));
        assert_eq!(entry.table, Some("nodes".to_string()));
        assert_eq!(entry.key, Some(b"key".to_vec()));
        assert_eq!(entry.value, Some(b"value".to_vec()));
        assert!(entry.is_data_operation());
        assert!(!entry.is_txn_marker());
    }

    #[test]
    fn test_delete_entry() {
        let entry = WalEntry::delete(2, "edges", b"edge_key");
        assert_eq!(entry.lsn, 2);
        assert!(matches!(entry.operation, Operation::Delete));
        assert!(entry.value.is_none());
        assert!(entry.is_data_operation());
    }

    #[test]
    fn test_txn_entries() {
        let begin = WalEntry::begin_txn(1, 100);
        let put = WalEntry::put_in_txn(2, 100, "nodes", b"k", b"v");
        let commit = WalEntry::commit_txn(3, 100);

        assert!(begin.is_txn_marker());
        assert!(!put.is_txn_marker());
        assert!(commit.is_txn_marker());

        assert_eq!(begin.txn_id, 100);
        assert_eq!(put.txn_id, 100);
        assert_eq!(commit.txn_id, 100);
    }

    #[test]
    fn test_checkpoint_entry() {
        let ckpt = WalEntry::checkpoint(100, 50);
        assert!(ckpt.is_checkpoint());
        assert_eq!(ckpt.checkpoint_lsn(), Some(50));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let entries = vec![
            WalEntry::put(1, "table", b"key", b"value"),
            WalEntry::delete(2, "table", b"key"),
            WalEntry::begin_txn(3, 42),
            WalEntry::commit_txn(4, 42),
            WalEntry::checkpoint(5, 3),
        ];

        for entry in entries {
            let serialized = bincode::serialize(&entry).unwrap();
            let deserialized: WalEntry = bincode::deserialize(&serialized).unwrap();
            assert_eq!(entry, deserialized);
        }
    }
}
