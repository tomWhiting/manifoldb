//! Product Quantization for vector compression.
//!
//! This module provides Product Quantization (PQ) for compressing high-dimensional
//! vectors into compact codes while preserving approximate distance computation.
//!
//! # Overview
//!
//! Product Quantization works by:
//! 1. Splitting vectors into M subspaces (segments)
//! 2. Training K centroids (codebook entries) per subspace using k-means
//! 3. Encoding each vector as M indices into the codebooks
//!
//! For example, a 128-dimensional vector with M=8 segments and K=256 centroids
//! can be compressed from 512 bytes (128 × 4 bytes) to just 8 bytes (8 × 1 byte).
//!
//! # Distance Computation
//!
//! Two modes are supported:
//! - **Symmetric Distance Computation (SDC)**: Approximate distances between two
//!   compressed vectors. Fast but less accurate.
//! - **Asymmetric Distance Computation (ADC)**: Exact query vector, compressed
//!   database vectors. Better accuracy for search.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::quantization::{ProductQuantizer, PQConfig};
//!
//! // Create a PQ with 8 segments and 256 centroids per segment
//! let config = PQConfig::new(128, 8).with_num_centroids(256);
//! let training_data: Vec<Vec<f32>> = /* your training vectors */;
//! let pq = ProductQuantizer::train(&config, &training_data)?;
//!
//! // Encode vectors
//! let vector = vec![0.1f32; 128];
//! let code = pq.encode(&vector);
//!
//! // Compute distances with ADC
//! let query = vec![0.2f32; 128];
//! let distance_table = pq.compute_distance_table(&query);
//! let distance = pq.asymmetric_distance(&distance_table, &code);
//! ```

mod config;
mod pq;
mod training;

pub use config::PQConfig;
pub use pq::{PQCode, ProductQuantizer};
pub use training::{KMeans, KMeansConfig};
