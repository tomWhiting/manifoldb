//! Graph analytics algorithms.
//!
//! This module provides common graph analytics algorithms for analyzing
//! graph structure, node importance, and community detection.
//!
//! # Algorithms
//!
//! - [`PageRank`] - Iterative power method for node importance ranking
//! - [`BetweennessCentrality`] - Brandes algorithm for node centrality
//! - [`CommunityDetection`] - Label Propagation for community detection
//! - [`ConnectedComponents`] - Weakly and strongly connected components
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{PageRank, PageRankConfig};
//!
//! // Run PageRank on the graph
//! let config = PageRankConfig::default();
//! let scores = PageRank::compute(&tx, &config)?;
//!
//! // Scores are returned as (EntityId, f64) pairs
//! for (node, score) in scores.iter().take(10) {
//!     println!("Node {:?} has PageRank score {:.4}", node, score);
//! }
//! ```

mod centrality;
mod community;
mod connected;
mod pagerank;

pub use centrality::{BetweennessCentrality, BetweennessCentralityConfig, CentralityResult};
pub use community::{CommunityDetection, CommunityDetectionConfig, CommunityResult};
pub use connected::{ComponentResult, ConnectedComponents, ConnectedComponentsConfig};
pub use pagerank::{PageRank, PageRankConfig, PageRankResult};
