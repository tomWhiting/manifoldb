//! Integration tests for ManifoldDB.
//!
//! This module contains comprehensive integration tests that exercise
//! the database across multiple features and at various scales.

pub mod combined;
pub mod crud;
pub mod ddl;
pub mod graph;
pub mod sql;
pub mod transactions;
pub mod vector;
