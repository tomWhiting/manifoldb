//! Graph analytics algorithms.
//!
//! This module provides common graph analytics algorithms for analyzing
//! graph structure, node importance, and community detection.
//!
//! # Algorithms
//!
//! ## Centrality Measures
//!
//! - [`PageRank`] - Iterative power method for node importance ranking
//! - [`BetweennessCentrality`] - Brandes algorithm for bridge/bottleneck detection
//! - [`DegreeCentrality`] - Simple degree-based importance (in/out/total)
//! - [`ClosenessCentrality`] - Distance-based centrality (standard and harmonic)
//! - [`EigenvectorCentrality`] - Importance based on connections to important nodes
//!
//! ## Community Detection
//!
//! - [`CommunityDetection`] - Label Propagation for community detection
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
//!
//! # Centrality Comparison
//!
//! | Algorithm | Measures | Time Complexity | Best For |
//! |-----------|----------|-----------------|----------|
//! | PageRank | Link importance | O(E * iterations) | Directed graphs, authority |
//! | Betweenness | Bridge nodes | O(V * E) | Finding bottlenecks |
//! | Degree | Direct connections | O(V) | Quick overview |
//! | Closeness | Path distances | O(V * (V + E)) | Finding central hubs |
//! | Eigenvector | Recursive importance | O(E * iterations) | Influence networks |

mod centrality;
mod closeness;
mod community;
mod degree;
mod eigenvector;
mod pagerank;

pub use centrality::{BetweennessCentrality, BetweennessCentralityConfig, CentralityResult};
pub use closeness::{ClosenessCentrality, ClosenessCentralityConfig, ClosenessCentralityResult};
pub use community::{CommunityDetection, CommunityDetectionConfig, CommunityResult};
pub use degree::{DegreeCentrality, DegreeCentralityConfig, DegreeCentralityResult};
pub use eigenvector::{
    EigenvectorCentrality, EigenvectorCentralityConfig, EigenvectorCentralityResult,
};
pub use pagerank::{PageRank, PageRankConfig, PageRankResult};
