//! `ManifoldDB` Query
//!
//! This crate provides query parsing, planning, and execution for `ManifoldDB`.
//!
//! # Overview
//!
//! The query system consists of several layers:
//!
//! - **AST**: Abstract syntax tree representation of parsed queries
//! - **Parser**: SQL parser with graph and vector extensions
//! - **Plan**: Query planning (logical and physical plans)
//! - **Exec**: Query execution engine
//!
//! # Modules
//!
//! - [`ast`] - Query abstract syntax tree types
//! - [`parser`] - SQL parser with graph/vector extensions
//! - [`plan`] - Query planning (logical and physical)
//! - [`exec`] - Query execution
//! - [`error`] - Error types for parsing and execution
//!
//! # Quick Start
//!
//! Parse a simple SQL query:
//!
//! ```
//! use manifoldb_query::parser::parse_sql;
//!
//! let statements = parse_sql("SELECT * FROM users WHERE id = 1").unwrap();
//! ```
//!
//! Parse a query with graph patterns:
//!
//! ```
//! use manifoldb_query::parser::ExtendedParser;
//!
//! let statements = ExtendedParser::parse(
//!     "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1"
//! ).unwrap();
//! ```
//!
//! Build AST programmatically:
//!
//! ```
//! use manifoldb_query::ast::{
//!     Expr, QualifiedName, SelectItem, SelectStatement, TableRef,
//! };
//!
//! let query = SelectStatement::new(vec![SelectItem::Wildcard])
//!     .from(TableRef::table(QualifiedName::simple("users")))
//!     .where_clause(
//!         Expr::column(QualifiedName::simple("id"))
//!             .eq(Expr::integer(1))
//!     );
//! ```

pub mod ast;
pub mod error;
pub mod exec;
pub mod parser;
pub mod plan;

// Re-export commonly used items at the crate root
pub use error::{ParseError, ParseResult};
pub use parser::{parse_single_statement, parse_sql, ExtendedParser};
