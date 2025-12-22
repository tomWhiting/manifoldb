//! Graph analytics operators for query execution.
//!
//! This module provides operators for running graph analytics algorithms
//! during query execution. These operators support the `CALL` syntax for
//! invoking analytics procedures:
//!
//! - `CALL pagerank(nodes) YIELD node, score`
//! - `CALL betweenness_centrality(nodes) YIELD node, score`
//! - `CALL community_detection(nodes) YIELD node, community`
//!
//! # Example
//!
//! ```ignore
//! // Create a PageRank operator
//! let config = PageRankOpConfig::default().with_damping_factor(0.85);
//! let op = PageRankOp::new(config);
//!
//! // Execute
//! op.open(&ctx)?;
//! while let Some(row) = op.next()? {
//!     // Each row contains: (node: EntityId, score: Float)
//!     println!("Node {:?} has score {:?}", row.get(0), row.get(1));
//! }
//! op.close()?;
//! ```

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::analytics::{
    BetweennessCentrality, BetweennessCentralityConfig, CommunityDetection,
    CommunityDetectionConfig, PageRank, PageRankConfig,
};
use manifoldb_graph::traversal::Direction;
use manifoldb_storage::Transaction;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};

/// Configuration for PageRank operator.
#[derive(Debug, Clone)]
pub struct PageRankOpConfig {
    /// Damping factor for PageRank algorithm.
    pub damping_factor: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Whether to normalize scores.
    pub normalize: bool,
}

impl Default for PageRankOpConfig {
    fn default() -> Self {
        let pr_config = PageRankConfig::default();
        Self {
            damping_factor: pr_config.damping_factor,
            max_iterations: pr_config.max_iterations,
            tolerance: pr_config.tolerance,
            normalize: pr_config.normalize,
        }
    }
}

impl PageRankOpConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the damping factor.
    pub const fn with_damping_factor(mut self, d: f64) -> Self {
        self.damping_factor = d;
        self
    }

    /// Set maximum iterations.
    pub const fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set convergence tolerance.
    pub const fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Set whether to normalize scores.
    pub const fn with_normalize(mut self, n: bool) -> Self {
        self.normalize = n;
        self
    }

    fn to_pagerank_config(&self) -> PageRankConfig {
        PageRankConfig::new()
            .with_damping_factor(self.damping_factor)
            .with_max_iterations(self.max_iterations)
            .with_tolerance(self.tolerance)
            .with_normalize(self.normalize)
    }
}

/// PageRank operator for computing node importance.
///
/// This operator computes PageRank scores for all nodes in the graph
/// and produces rows with (node_id, score) pairs.
///
/// # Output Schema
///
/// - `node`: The entity ID (as integer)
/// - `score`: The PageRank score (as float)
pub struct PageRankOp<T> {
    /// Operator base with schema and state.
    base: OperatorBase,
    /// Configuration for PageRank.
    config: PageRankOpConfig,
    /// Transaction for graph access.
    tx: Option<T>,
    /// Results iterator.
    results: Option<std::vec::IntoIter<(EntityId, f64)>>,
    /// Optional input operator providing nodes to analyze.
    input: Option<BoxedOperator>,
    /// Column index for node input (if using input operator).
    input_node_column: Option<usize>,
}

impl<T> PageRankOp<T> {
    /// Create a new PageRank operator.
    ///
    /// This variant computes PageRank on all nodes in the graph.
    pub fn new(config: PageRankOpConfig) -> Self {
        let schema = Arc::new(Schema::new(vec!["node".to_string(), "score".to_string()]));
        Self {
            base: OperatorBase::new(schema),
            config,
            tx: None,
            results: None,
            input: None,
            input_node_column: None,
        }
    }

    /// Create a PageRank operator with an input operator.
    ///
    /// This variant computes PageRank only for nodes produced by the input.
    pub fn with_input(config: PageRankOpConfig, input: BoxedOperator, node_column: usize) -> Self {
        let schema = Arc::new(Schema::new(vec!["node".to_string(), "score".to_string()]));
        Self {
            base: OperatorBase::new(schema),
            config,
            tx: None,
            results: None,
            input: Some(input),
            input_node_column: Some(node_column),
        }
    }

    /// Set the transaction to use.
    pub fn with_tx(mut self, tx: T) -> Self {
        self.tx = Some(tx);
        self
    }
}

impl<T> Operator for PageRankOp<T>
where
    T: Transaction + Send,
{
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Open input if present
        if let Some(ref mut input) = self.input {
            input.open(ctx)?;
        }

        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // If we haven't computed results yet, do so now
        if self.results.is_none() {
            let tx = self.tx.as_ref().ok_or_else(|| {
                ParseError::InvalidGraphOp("PageRank requires transaction access".to_string())
            })?;

            let pr_config = self.config.to_pagerank_config();

            // Collect input nodes if we have an input operator
            let result = if let Some(ref mut input) = self.input {
                let column = self.input_node_column.unwrap_or(0);
                let mut nodes = Vec::new();

                while let Some(row) = input.next()? {
                    if let Some(Value::Int(id)) = row.get(column) {
                        nodes.push(EntityId::new(*id as u64));
                    }
                }

                PageRank::compute_for_nodes(tx, &nodes, &pr_config)
                    .map_err(|e| ParseError::InvalidGraphOp(format!("PageRank error: {e}")))?
            } else {
                PageRank::compute(tx, &pr_config)
                    .map_err(|e| ParseError::InvalidGraphOp(format!("PageRank error: {e}")))?
            };

            // Sort results by score (descending) for consistent output
            let sorted = result.sorted();
            self.results = Some(sorted.into_iter());
        }

        // Return next result
        if let Some(ref mut iter) = self.results {
            if let Some((node, score)) = iter.next() {
                let row = Row::new(
                    self.base.schema(),
                    vec![Value::Int(node.as_u64() as i64), Value::Float(score)],
                );
                self.base.inc_rows_produced();
                return Ok(Some(row));
            }
        }

        self.base.set_finished();
        Ok(None)
    }

    fn close(&mut self) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.close()?;
        }
        self.results = None;
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "PageRankOp"
    }
}

/// Configuration for Betweenness Centrality operator.
#[derive(Debug, Clone)]
pub struct BetweennessCentralityOpConfig {
    /// Whether to normalize centrality values.
    pub normalize: bool,
    /// Direction of edges to follow.
    pub direction: Direction,
}

impl Default for BetweennessCentralityOpConfig {
    fn default() -> Self {
        let bc_config = BetweennessCentralityConfig::default();
        Self { normalize: bc_config.normalize, direction: bc_config.direction }
    }
}

impl BetweennessCentralityOpConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to normalize centrality values.
    pub const fn with_normalize(mut self, n: bool) -> Self {
        self.normalize = n;
        self
    }

    /// Set the direction to follow edges.
    pub const fn with_direction(mut self, d: Direction) -> Self {
        self.direction = d;
        self
    }

    fn to_centrality_config(&self) -> BetweennessCentralityConfig {
        BetweennessCentralityConfig::new()
            .with_normalize(self.normalize)
            .with_direction(self.direction)
    }
}

/// Betweenness Centrality operator.
///
/// This operator computes betweenness centrality for all nodes in the graph
/// and produces rows with (node_id, score) pairs.
///
/// # Output Schema
///
/// - `node`: The entity ID (as integer)
/// - `score`: The centrality score (as float)
pub struct BetweennessCentralityOp<T> {
    /// Operator base with schema and state.
    base: OperatorBase,
    /// Configuration for centrality.
    config: BetweennessCentralityOpConfig,
    /// Transaction for graph access.
    tx: Option<T>,
    /// Results iterator.
    results: Option<std::vec::IntoIter<(EntityId, f64)>>,
    /// Optional input operator.
    input: Option<BoxedOperator>,
    /// Column index for node input.
    input_node_column: Option<usize>,
}

impl<T> BetweennessCentralityOp<T> {
    /// Create a new Betweenness Centrality operator.
    pub fn new(config: BetweennessCentralityOpConfig) -> Self {
        let schema = Arc::new(Schema::new(vec!["node".to_string(), "score".to_string()]));
        Self {
            base: OperatorBase::new(schema),
            config,
            tx: None,
            results: None,
            input: None,
            input_node_column: None,
        }
    }

    /// Create with an input operator.
    pub fn with_input(
        config: BetweennessCentralityOpConfig,
        input: BoxedOperator,
        node_column: usize,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec!["node".to_string(), "score".to_string()]));
        Self {
            base: OperatorBase::new(schema),
            config,
            tx: None,
            results: None,
            input: Some(input),
            input_node_column: Some(node_column),
        }
    }

    /// Set the transaction to use.
    pub fn with_tx(mut self, tx: T) -> Self {
        self.tx = Some(tx);
        self
    }
}

impl<T> Operator for BetweennessCentralityOp<T>
where
    T: Transaction + Send,
{
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.open(ctx)?;
        }
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.results.is_none() {
            let tx = self.tx.as_ref().ok_or_else(|| {
                ParseError::InvalidGraphOp(
                    "BetweennessCentrality requires transaction access".to_string(),
                )
            })?;

            let bc_config = self.config.to_centrality_config();

            let result = if let Some(ref mut input) = self.input {
                let column = self.input_node_column.unwrap_or(0);
                let mut nodes = Vec::new();

                while let Some(row) = input.next()? {
                    if let Some(Value::Int(id)) = row.get(column) {
                        nodes.push(EntityId::new(*id as u64));
                    }
                }

                BetweennessCentrality::compute_for_nodes(tx, &nodes, &bc_config).map_err(|e| {
                    ParseError::InvalidGraphOp(format!("BetweennessCentrality error: {e}"))
                })?
            } else {
                BetweennessCentrality::compute(tx, &bc_config).map_err(|e| {
                    ParseError::InvalidGraphOp(format!("BetweennessCentrality error: {e}"))
                })?
            };

            let sorted = result.sorted();
            self.results = Some(sorted.into_iter());
        }

        if let Some(ref mut iter) = self.results {
            if let Some((node, score)) = iter.next() {
                let row = Row::new(
                    self.base.schema(),
                    vec![Value::Int(node.as_u64() as i64), Value::Float(score)],
                );
                self.base.inc_rows_produced();
                return Ok(Some(row));
            }
        }

        self.base.set_finished();
        Ok(None)
    }

    fn close(&mut self) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.close()?;
        }
        self.results = None;
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "BetweennessCentralityOp"
    }
}

/// Configuration for Community Detection operator.
#[derive(Debug, Clone)]
pub struct CommunityDetectionOpConfig {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Direction of edges to follow.
    pub direction: Direction,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for CommunityDetectionOpConfig {
    fn default() -> Self {
        let cd_config = CommunityDetectionConfig::default();
        Self {
            max_iterations: cd_config.max_iterations,
            direction: cd_config.direction,
            seed: cd_config.seed,
        }
    }
}

impl CommunityDetectionOpConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum iterations.
    pub const fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the direction to follow edges.
    pub const fn with_direction(mut self, d: Direction) -> Self {
        self.direction = d;
        self
    }

    /// Set the seed for reproducibility.
    pub const fn with_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    fn to_community_config(&self) -> CommunityDetectionConfig {
        let mut config = CommunityDetectionConfig::new()
            .with_max_iterations(self.max_iterations)
            .with_direction(self.direction);
        if let Some(seed) = self.seed {
            config = config.with_seed(seed);
        }
        config
    }
}

/// Community Detection operator.
///
/// This operator detects communities using Label Propagation and produces
/// rows with (node_id, community_id) pairs.
///
/// # Output Schema
///
/// - `node`: The entity ID (as integer)
/// - `community`: The community ID (as integer)
pub struct CommunityDetectionOp<T> {
    /// Operator base with schema and state.
    base: OperatorBase,
    /// Configuration for community detection.
    config: CommunityDetectionOpConfig,
    /// Transaction for graph access.
    tx: Option<T>,
    /// Results iterator.
    results: Option<std::vec::IntoIter<(EntityId, u64)>>,
    /// Optional input operator.
    input: Option<BoxedOperator>,
    /// Column index for node input.
    input_node_column: Option<usize>,
}

impl<T> CommunityDetectionOp<T> {
    /// Create a new Community Detection operator.
    pub fn new(config: CommunityDetectionOpConfig) -> Self {
        let schema = Arc::new(Schema::new(vec!["node".to_string(), "community".to_string()]));
        Self {
            base: OperatorBase::new(schema),
            config,
            tx: None,
            results: None,
            input: None,
            input_node_column: None,
        }
    }

    /// Create with an input operator.
    pub fn with_input(
        config: CommunityDetectionOpConfig,
        input: BoxedOperator,
        node_column: usize,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec!["node".to_string(), "community".to_string()]));
        Self {
            base: OperatorBase::new(schema),
            config,
            tx: None,
            results: None,
            input: Some(input),
            input_node_column: Some(node_column),
        }
    }

    /// Set the transaction to use.
    pub fn with_tx(mut self, tx: T) -> Self {
        self.tx = Some(tx);
        self
    }
}

impl<T> Operator for CommunityDetectionOp<T>
where
    T: Transaction + Send,
{
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.open(ctx)?;
        }
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.results.is_none() {
            let tx = self.tx.as_ref().ok_or_else(|| {
                ParseError::InvalidGraphOp(
                    "CommunityDetection requires transaction access".to_string(),
                )
            })?;

            let cd_config = self.config.to_community_config();

            let result = if let Some(ref mut input) = self.input {
                let column = self.input_node_column.unwrap_or(0);
                let mut nodes = Vec::new();

                while let Some(row) = input.next()? {
                    if let Some(Value::Int(id)) = row.get(column) {
                        nodes.push(EntityId::new(*id as u64));
                    }
                }

                CommunityDetection::label_propagation_for_nodes(tx, &nodes, &cd_config).map_err(
                    |e| ParseError::InvalidGraphOp(format!("CommunityDetection error: {e}")),
                )?
            } else {
                CommunityDetection::label_propagation(tx, &cd_config).map_err(|e| {
                    ParseError::InvalidGraphOp(format!("CommunityDetection error: {e}"))
                })?
            };

            // Convert assignments to sorted list by node ID
            let mut pairs: Vec<_> = result.assignments.into_iter().collect();
            pairs.sort_by_key(|(id, _)| id.as_u64());
            self.results = Some(pairs.into_iter());
        }

        if let Some(ref mut iter) = self.results {
            if let Some((node, community)) = iter.next() {
                let row = Row::new(
                    self.base.schema(),
                    vec![Value::Int(node.as_u64() as i64), Value::Int(community as i64)],
                );
                self.base.inc_rows_produced();
                return Ok(Some(row));
            }
        }

        self.base.set_finished();
        Ok(None)
    }

    fn close(&mut self) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.close()?;
        }
        self.results = None;
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "CommunityDetectionOp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pagerank_config_defaults() {
        let config = PageRankOpConfig::default();
        assert!((config.damping_factor - 0.85).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 100);
        assert!(config.normalize);
    }

    #[test]
    fn pagerank_config_builder() {
        let config = PageRankOpConfig::new()
            .with_damping_factor(0.9)
            .with_max_iterations(50)
            .with_tolerance(1e-8)
            .with_normalize(false);

        assert!((config.damping_factor - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 50);
        assert!((config.tolerance - 1e-8).abs() < f64::EPSILON);
        assert!(!config.normalize);
    }

    #[test]
    fn betweenness_centrality_config_defaults() {
        let config = BetweennessCentralityOpConfig::default();
        assert!(config.normalize);
        assert_eq!(config.direction, Direction::Both);
    }

    #[test]
    fn community_detection_config_defaults() {
        let config = CommunityDetectionOpConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.direction, Direction::Both);
        assert!(config.seed.is_none());
    }

    #[test]
    fn community_detection_config_builder() {
        let config = CommunityDetectionOpConfig::new()
            .with_max_iterations(50)
            .with_direction(Direction::Outgoing)
            .with_seed(42);

        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.direction, Direction::Outgoing);
        assert_eq!(config.seed, Some(42));
    }
}
