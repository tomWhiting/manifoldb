//! SQL parser with graph and vector extensions.
//!
//! This module provides parsing of SQL queries with custom syntax
//! for graph traversal and vector similarity search.
//!
//! # Overview
//!
//! The parser is built on top of [`sqlparser-rs`](https://crates.io/crates/sqlparser)
//! and extends it with custom syntax for:
//!
//! - **Graph patterns**: MATCH clauses using Cypher-like syntax
//! - **Vector operations**: Distance operators (<->, <=>, <#>)
//!
//! # Usage
//!
//! For standard SQL queries, use [`parse_sql`]:
//!
//! ```ignore
//! use manifoldb_query::parser::parse_sql;
//!
//! let stmts = parse_sql("SELECT * FROM users WHERE id = 1")?;
//! ```
//!
//! For extended queries with graph/vector syntax, use [`ExtendedParser`]:
//!
//! ```ignore
//! use manifoldb_query::parser::ExtendedParser;
//!
//! let stmts = ExtendedParser::parse(
//!     "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1"
//! )?;
//! ```
//!
//! # Extended Syntax
//!
//! ## Vector Distance Operators
//!
//! ```sql
//! -- Euclidean distance (L2)
//! SELECT * FROM docs ORDER BY embedding <-> $query LIMIT 10;
//!
//! -- Cosine distance
//! SELECT * FROM docs WHERE embedding <=> $query < 0.5;
//!
//! -- Inner product (negative for similarity)
//! SELECT * FROM docs ORDER BY embedding <#> $query DESC LIMIT 10;
//! ```
//!
//! ## Graph Pattern Matching
//!
//! ```sql
//! -- Find followers
//! SELECT f.* FROM users u
//! MATCH (u)-[:FOLLOWS]->(f)
//! WHERE u.id = 1;
//!
//! -- Multi-hop paths
//! SELECT * FROM users
//! MATCH (a)-[:KNOWS*1..3]->(b)
//! WHERE a.id = 1;
//!
//! -- Combined graph and vector
//! SELECT d.*, a.name
//! FROM docs d
//! MATCH (d)-[:AUTHORED_BY]->(a)
//! WHERE d.embedding <-> $query < 0.5;
//! ```

pub mod extensions;
pub mod sql;

// Re-export commonly used items
pub use extensions::ExtendedParser;
pub use sql::{parse_check_expression, parse_single_statement, parse_sql};
