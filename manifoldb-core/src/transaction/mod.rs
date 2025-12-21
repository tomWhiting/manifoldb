//! Transaction types and error handling for `ManifoldDB`.
//!
//! This module provides the core transaction error types and traits that are
//! shared across the `ManifoldDB` crates. The concrete `TransactionManager` and
//! `DatabaseTransaction` implementations live in the main `manifoldb` crate.
//!
//! # Architecture
//!
//! The transaction system follows a layered design:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    TransactionManager                        │
//! │  - Coordinates all stores (in manifoldb crate)              │
//! │  - Owns storage engine                                       │
//! │  - Manages transaction lifecycle                             │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   DatabaseTransaction                        │
//! │  - User-facing transaction handle (in manifoldb crate)      │
//! │  - Wraps storage transaction                                 │
//! │  - Provides graph/vector operations                          │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StorageTransaction                        │
//! │  - Low-level key-value operations (in manifoldb-storage)    │
//! │  - Backend-specific implementation                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Transaction Semantics
//!
//! - **Read transactions**: Snapshot isolation, multiple concurrent readers
//! - **Write transactions**: Serializable, single writer at a time (initially)
//! - All index updates occur within the same transaction as data
//!
//! # Vector Index Consistency
//!
//! The transaction manager supports multiple consistency strategies:
//!
//! - **Synchronous**: Update vector indexes in the same transaction (default)
//! - **Async**: Queue updates for background processing (future optimization)
//! - **Hybrid**: Synchronous for small batches, async for bulk operations
//!
//! The default is synchronous for correctness; optimization can be done later.

mod error;

pub use error::{TransactionError, TransactionErrorContext, TransactionResult};
