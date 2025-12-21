//! Query execution engine.
//!
//! This module provides the query executor and operators for running
//! physical query plans against the database.
//!
//! # Architecture
//!
//! The execution engine uses a **pull-based iterator model** where each
//! operator implements the [`Operator`] trait with `open()`, `next()`,
//! and `close()` methods. Data flows from bottom to top of the plan tree.
//!
//! # Modules
//!
//! - [`context`] - Execution context (transactions, parameters)
//! - [`row`] - Row type for intermediate results
//! - [`operator`] - Operator trait and base types
//! - [`operators`] - Concrete operator implementations
//! - [`result`] - Query result types
//! - [`executor`] - Main executor that drives query execution
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_query::exec::{Executor, ExecutionContext};
//!
//! let ctx = ExecutionContext::new(&transaction);
//! let mut executor = Executor::new(&plan, &ctx)?;
//! while let Some(row) = executor.next()? {
//!     println!("{:?}", row);
//! }
//! ```

mod context;
mod executor;
pub mod graph_accessor;
mod operator;
mod result;

pub mod operators;
pub mod row;

// Re-exports
pub use context::{CancellationToken, ExecutionContext};
pub use executor::{execute_plan, Executor};
pub use graph_accessor::{
    GraphAccessError, GraphAccessResult, GraphAccessor, NeighborResult, NullGraphAccessor,
    TransactionGraphAccessor, TraversalResult,
};
pub use operator::{Operator, OperatorBase, OperatorState};
pub use result::{QueryResult, ResultSet, ResultSetBuilder};
pub use row::{Row, RowBatch, Schema};
