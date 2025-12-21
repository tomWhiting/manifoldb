//! `ManifoldDB` Query
//!
//! This crate provides query parsing, planning, and execution for `ManifoldDB`.
//!
//! # Modules
//!
//! - [`ast`] - Query abstract syntax tree
//! - [`parser`] - SQL parser with extensions
//! - [`plan`] - Query planning (logical and physical)
//! - [`exec`] - Query execution

pub mod ast;
pub mod exec;
pub mod parser;
pub mod plan;
