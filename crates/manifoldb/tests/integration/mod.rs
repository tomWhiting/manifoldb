//! Integration tests for ManifoldDB.
//!
//! This module contains comprehensive integration tests that exercise
//! the database across multiple features and at various scales.

pub mod bulk_vectors;
pub mod combined;
pub mod concurrency;
pub mod correctness;
pub mod crud;
pub mod ddl;
pub mod e2e;
pub mod graph;
pub mod index_maintenance;
pub mod prepared;
pub mod recovery;
pub mod scale;
pub mod sql;
pub mod transactions;
pub mod vector;
