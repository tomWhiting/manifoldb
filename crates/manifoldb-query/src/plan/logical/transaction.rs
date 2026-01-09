//! Transaction control plan nodes.
//!
//! This module defines logical plan nodes for transaction control operations:
//! - BEGIN / START TRANSACTION
//! - COMMIT
//! - ROLLBACK
//! - SAVEPOINT
//! - RELEASE SAVEPOINT
//! - SET TRANSACTION

use crate::ast::{IsolationLevel, TransactionAccessMode};

/// A BEGIN TRANSACTION operation.
///
/// Starts a new transaction with optional configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BeginTransactionNode {
    /// Transaction isolation level.
    pub isolation_level: Option<IsolationLevel>,
    /// Transaction access mode.
    pub access_mode: Option<TransactionAccessMode>,
    /// Whether this is a deferred transaction.
    pub deferred: bool,
}

impl BeginTransactionNode {
    /// Creates a new BEGIN TRANSACTION node with default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the isolation level.
    #[must_use]
    pub fn with_isolation_level(mut self, level: IsolationLevel) -> Self {
        self.isolation_level = Some(level);
        self
    }

    /// Sets the access mode.
    #[must_use]
    pub fn with_access_mode(mut self, mode: TransactionAccessMode) -> Self {
        self.access_mode = Some(mode);
        self
    }

    /// Sets the deferred flag.
    #[must_use]
    pub fn deferred(mut self) -> Self {
        self.deferred = true;
        self
    }
}

/// A COMMIT operation.
///
/// Commits the current transaction, making all changes permanent.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommitNode;

impl CommitNode {
    /// Creates a new COMMIT node.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// A ROLLBACK operation.
///
/// Rolls back the current transaction, undoing all changes.
/// Optionally rolls back to a named savepoint.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RollbackNode {
    /// Optional savepoint name to roll back to.
    pub to_savepoint: Option<String>,
}

impl RollbackNode {
    /// Creates a ROLLBACK node that rolls back the entire transaction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a ROLLBACK TO SAVEPOINT node.
    #[must_use]
    pub fn to_savepoint(name: impl Into<String>) -> Self {
        Self { to_savepoint: Some(name.into()) }
    }
}

/// A SAVEPOINT operation.
///
/// Creates a savepoint within the current transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavepointNode {
    /// The savepoint name.
    pub name: String,
}

impl SavepointNode {
    /// Creates a new SAVEPOINT node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// A RELEASE SAVEPOINT operation.
///
/// Removes a savepoint without affecting the transaction state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseSavepointNode {
    /// The savepoint name to release.
    pub name: String,
}

impl ReleaseSavepointNode {
    /// Creates a new RELEASE SAVEPOINT node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// A SET TRANSACTION operation.
///
/// Changes the characteristics of the current transaction.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SetTransactionNode {
    /// Transaction isolation level.
    pub isolation_level: Option<IsolationLevel>,
    /// Transaction access mode.
    pub access_mode: Option<TransactionAccessMode>,
}

impl SetTransactionNode {
    /// Creates a new SET TRANSACTION node.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the isolation level.
    #[must_use]
    pub fn with_isolation_level(mut self, level: IsolationLevel) -> Self {
        self.isolation_level = Some(level);
        self
    }

    /// Sets the access mode.
    #[must_use]
    pub fn with_access_mode(mut self, mode: TransactionAccessMode) -> Self {
        self.access_mode = Some(mode);
        self
    }
}
