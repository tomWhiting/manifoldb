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
mod operator;
mod result;
mod row;

pub mod operators;

// Re-exports
pub use context::ExecutionContext;
pub use executor::Executor;
pub use operator::{Operator, OperatorState};
pub use result::{QueryResult, ResultSet};
pub use row::{Row, RowBatch};
