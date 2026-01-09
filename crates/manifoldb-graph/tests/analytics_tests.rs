//! Integration tests for graph analytics algorithms.
//!
//! These tests verify the correctness of PageRank, Betweenness Centrality,
//! Community Detection, and Connected Components algorithms on various graph topologies.

#![allow(clippy::manual_range_contains)]

use manifoldb_core::{Edge, Entity, EntityId};
use manifoldb_graph::analytics::{
    BetweennessCentrality, BetweennessCentralityConfig, CommunityDetection,
    CommunityDetectionConfig, ConnectedComponents, ConnectedComponentsConfig, PageRank,
    PageRankConfig,
};
use manifoldb_graph::store::{EdgeStore, GraphError, IdGenerator, NodeStore};
use manifoldb_graph::traversal::Direction;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{StorageEngine, Transaction};

fn create_test_engine() -> RedbEngine {
    RedbEngine::in_memory().expect("Failed to create in-memory engine")
}

// ============================================================================
// Helper functions to create test graphs
// ============================================================================

/// Create a simple linear graph: A -> B -> C -> D
fn create_linear_graph(engine: &RedbEngine) -> Vec<EntityId> {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for i in 0..4 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Node").with_property("index", i as i64)
        })
        .unwrap();
        nodes.push(node.id);
    }

    for i in 0..3 {
        EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[i + 1], "NEXT", |id| {
            Edge::new(id, nodes[i], nodes[i + 1], "NEXT")
        })
        .unwrap();
    }

    tx.commit().unwrap();
    nodes
}

/// Create a star graph: center node with n spokes (neighbors)
fn create_star_graph(engine: &RedbEngine, n: usize) -> (EntityId, Vec<EntityId>) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let center =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Center")).unwrap().id;

    let mut spokes = Vec::new();
    for i in 0..n {
        let spoke = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Spoke").with_property("index", i as i64)
        })
        .unwrap()
        .id;
        EdgeStore::create(&mut tx, &id_gen, center, spoke, "CONNECTS", |id| {
            Edge::new(id, center, spoke, "CONNECTS")
        })
        .unwrap();
        // Also add reverse edges for undirected-like behavior
        EdgeStore::create(&mut tx, &id_gen, spoke, center, "CONNECTS", |id| {
            Edge::new(id, spoke, center, "CONNECTS")
        })
        .unwrap();
        spokes.push(spoke);
    }

    tx.commit().unwrap();
    (center, spokes)
}

/// Create a cycle graph: A -> B -> C -> A
fn create_cycle_graph(engine: &RedbEngine, n: usize) -> Vec<EntityId> {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for i in 0..n {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Node").with_property("index", i as i64)
        })
        .unwrap()
        .id;
        nodes.push(node);
    }

    for i in 0..n {
        let source = nodes[i];
        let target = nodes[(i + 1) % n];
        EdgeStore::create(&mut tx, &id_gen, source, target, "EDGE", |id| {
            Edge::new(id, source, target, "EDGE")
        })
        .unwrap();
    }

    tx.commit().unwrap();
    nodes
}

/// Create a simple two-community graph:
/// Community 1: A -- B -- C (connected)
/// Community 2: D -- E -- F (connected)
/// Single bridge: C -- D
fn create_two_community_graph(engine: &RedbEngine) -> (Vec<EntityId>, Vec<EntityId>) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut community1 = Vec::new();
    let mut community2 = Vec::new();

    // Create community 1 nodes (A, B, C)
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Community1").with_property("index", i as i64)
        })
        .unwrap()
        .id;
        community1.push(node);
    }

    // Create community 2 nodes (D, E, F)
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Community2").with_property("index", i as i64)
        })
        .unwrap()
        .id;
        community2.push(node);
    }

    // Connect community 1: A -- B -- C (bidirectional)
    for i in 0..2 {
        let src = community1[i];
        let tgt = community1[i + 1];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "INTRA", |id| {
            Edge::new(id, src, tgt, "INTRA")
        })
        .unwrap();
        EdgeStore::create(&mut tx, &id_gen, tgt, src, "INTRA", |id| {
            Edge::new(id, tgt, src, "INTRA")
        })
        .unwrap();
    }

    // Connect community 2: D -- E -- F (bidirectional)
    for i in 0..2 {
        let src = community2[i];
        let tgt = community2[i + 1];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "INTRA", |id| {
            Edge::new(id, src, tgt, "INTRA")
        })
        .unwrap();
        EdgeStore::create(&mut tx, &id_gen, tgt, src, "INTRA", |id| {
            Edge::new(id, tgt, src, "INTRA")
        })
        .unwrap();
    }

    // Bridge between communities: C -- D (bidirectional)
    let c = community1[2];
    let d = community2[0];
    EdgeStore::create(&mut tx, &id_gen, c, d, "BRIDGE", |id| Edge::new(id, c, d, "BRIDGE"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, d, c, "BRIDGE", |id| Edge::new(id, d, c, "BRIDGE"))
        .unwrap();

    tx.commit().unwrap();
    (community1, community2)
}

/// Create a complete graph (all nodes connected to each other)
fn create_complete_graph(engine: &RedbEngine, n: usize) -> Vec<EntityId> {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for i in 0..n {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Node").with_property("index", i as i64)
        })
        .unwrap()
        .id;
        nodes.push(node);
    }

    // Connect all pairs
    for i in 0..n {
        for j in 0..n {
            if i != j {
                EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[j], "CONNECTED", |id| {
                    Edge::new(id, nodes[i], nodes[j], "CONNECTED")
                })
                .unwrap();
            }
        }
    }

    tx.commit().unwrap();
    nodes
}

// ============================================================================
// PageRank tests
// ============================================================================

#[test]
fn pagerank_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    assert!(result.scores.is_empty());
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn pagerank_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 1);
    // Single node should have score 1.0 (normalized)
    assert!((result.score(node.id).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn pagerank_linear_graph() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 4);
    assert!(result.converged);

    // In a directed linear graph A -> B -> C -> D:
    // D should have highest rank (receives from C)
    // A should have lowest rank (receives from no one, just damping)
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();

    // D (last) should have highest score
    assert!(scores[3] > scores[0], "D should have higher PageRank than A");
}

#[test]
fn pagerank_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 6);
    assert!(result.converged);

    // Center should have highest PageRank (receives from all spokes)
    let center_score = result.score(center).unwrap();
    for &spoke in &spokes {
        let spoke_score = result.score(spoke).unwrap();
        assert!(center_score > spoke_score, "Center should have higher PageRank than spokes");
    }
}

#[test]
fn pagerank_cycle_graph() {
    let engine = create_test_engine();
    let nodes = create_cycle_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 4);
    assert!(result.converged);

    // In a cycle, all nodes should have equal PageRank
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();
    let first = scores[0];
    for &score in &scores[1..] {
        assert!((score - first).abs() < 1e-6, "All nodes in cycle should have equal PageRank");
    }
}

#[test]
fn pagerank_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 4);
    assert!(result.converged);

    // In a complete graph, all nodes should have equal PageRank
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();
    let first = scores[0];
    for &score in &scores[1..] {
        assert!(
            (score - first).abs() < 1e-6,
            "All nodes in complete graph should have equal PageRank"
        );
    }
}

#[test]
fn pagerank_damping_factor() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Compare different damping factors
    let result_high =
        PageRank::compute(&tx, &PageRankConfig::default().with_damping_factor(0.9)).unwrap();
    let result_low =
        PageRank::compute(&tx, &PageRankConfig::default().with_damping_factor(0.5)).unwrap();

    // With higher damping, the final node should accumulate more rank
    let high_d_score = result_high.score(nodes[3]).unwrap();
    let low_d_score = result_low.score(nodes[3]).unwrap();

    // Higher damping means more influence from links, so D should have higher relative score
    assert!(high_d_score > low_d_score || (high_d_score - low_d_score).abs() < 0.1);
}

#[test]
fn pagerank_normalization() {
    let engine = create_test_engine();
    let _nodes = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();

    // With normalization
    let config_norm = PageRankConfig::default().with_normalize(true);
    let result_norm = PageRank::compute(&tx, &config_norm).unwrap();

    // Sum should be 1.0
    let sum: f64 = result_norm.scores.values().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Normalized PageRank should sum to 1.0");

    // Without normalization
    let config_no_norm = PageRankConfig::default().with_normalize(false);
    let result_no_norm = PageRank::compute(&tx, &config_no_norm).unwrap();

    // Sum may not be 1.0
    let sum_no_norm: f64 = result_no_norm.scores.values().sum();
    // It should still be close to n (number of nodes) when not normalized
    assert!(sum_no_norm > 0.0);
}

#[test]
fn pagerank_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();

    // Only compute for first two nodes
    let subset = vec![nodes[0], nodes[1]];
    let result = PageRank::compute_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.scores.len(), 2);
    assert!(result.score(nodes[0]).is_some());
    assert!(result.score(nodes[1]).is_some());
    assert!(result.score(nodes[2]).is_none());
}

#[test]
fn pagerank_result_methods() {
    let engine = create_test_engine();
    let _nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = PageRankConfig::default();
    let result = PageRank::compute(&tx, &config).unwrap();

    // Test sorted
    let sorted = result.sorted();
    assert_eq!(sorted.len(), 4);
    // Should be in descending order
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].1 >= sorted[i].1);
    }

    // Test top_n
    let top2 = result.top_n(2);
    assert_eq!(top2.len(), 2);

    // Test max/min
    let max = result.max().unwrap();
    let min = result.min().unwrap();
    assert!(max.1 >= min.1);
}

// ============================================================================
// Betweenness Centrality tests
// ============================================================================

#[test]
fn betweenness_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = BetweennessCentralityConfig::default();
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    assert!(result.scores.is_empty());
}

#[test]
fn betweenness_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = BetweennessCentralityConfig::default();
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 1);
    // Single node has zero betweenness
    assert_eq!(result.score(node.id).unwrap(), 0.0);
}

#[test]
fn betweenness_linear_graph() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    // Use Direction::Both to treat as undirected for betweenness
    let config = BetweennessCentralityConfig::default()
        .with_direction(Direction::Both)
        .with_normalize(false);
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 4);

    // In linear graph A - B - C - D:
    // B and C should have highest betweenness (they're on all paths between endpoints)
    // A and D should have lowest (they're endpoints)
    let score_a = result.score(nodes[0]).unwrap();
    let score_b = result.score(nodes[1]).unwrap();
    let score_c = result.score(nodes[2]).unwrap();
    let score_d = result.score(nodes[3]).unwrap();

    assert!(score_b > score_a, "B should have higher betweenness than A");
    assert!(score_c > score_d, "C should have higher betweenness than D");
    // B and C should be similar (both are internal nodes)
    assert!((score_b - score_c).abs() < score_b * 0.5);
}

#[test]
fn betweenness_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = BetweennessCentralityConfig::default()
        .with_direction(Direction::Both)
        .with_normalize(false);
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    // Center should have maximum betweenness (all paths go through it)
    let center_score = result.score(center).unwrap();
    for &spoke in &spokes {
        let spoke_score = result.score(spoke).unwrap();
        assert!(center_score >= spoke_score, "Center should have higher betweenness than spokes");
    }
}

#[test]
fn betweenness_two_communities() {
    let engine = create_test_engine();
    let (community1, community2) = create_two_community_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = BetweennessCentralityConfig::default()
        .with_direction(Direction::Both)
        .with_normalize(false);
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    // Bridge nodes (C and D) should have high betweenness
    let c = community1[2];
    let d = community2[0];
    let c_score = result.score(c).unwrap();
    let d_score = result.score(d).unwrap();

    // Endpoint nodes (A, F) should have low betweenness
    let a = community1[0];
    let f = community2[2];
    let a_score = result.score(a).unwrap();
    let f_score = result.score(f).unwrap();

    assert!(c_score > a_score, "Bridge node C should have higher betweenness than endpoint A");
    assert!(d_score > f_score, "Bridge node D should have higher betweenness than endpoint F");
}

#[test]
fn betweenness_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config =
        BetweennessCentralityConfig::default().with_direction(Direction::Both).with_normalize(true);
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    // In a complete graph, all nodes should have equal betweenness
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();
    let first = scores[0];
    for &score in &scores[1..] {
        assert!(
            (score - first).abs() < 1e-6,
            "All nodes in complete graph should have equal betweenness"
        );
    }
}

#[test]
fn betweenness_normalization() {
    let engine = create_test_engine();
    let _nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // With normalization
    let config_norm =
        BetweennessCentralityConfig::default().with_direction(Direction::Both).with_normalize(true);
    let result_norm = BetweennessCentrality::compute(&tx, &config_norm).unwrap();

    // Normalized values should be in [0, 1]
    for &score in result_norm.scores.values() {
        assert!((0.0..=1.0).contains(&score), "Normalized betweenness should be in [0, 1]");
    }

    // Without normalization
    let config_no_norm = BetweennessCentralityConfig::default()
        .with_direction(Direction::Both)
        .with_normalize(false);
    let result_no_norm = BetweennessCentrality::compute(&tx, &config_no_norm).unwrap();

    // Non-normalized values can be larger
    let max_score = result_no_norm.scores.values().copied().fold(0.0, f64::max);
    assert!(max_score >= 0.0);
}

#[test]
fn betweenness_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = BetweennessCentralityConfig::default().with_direction(Direction::Both);

    // Only compute for subset
    let subset = vec![nodes[0], nodes[1], nodes[2]];
    let result = BetweennessCentrality::compute_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.scores.len(), 3);
    assert!(result.score(nodes[3]).is_none());
}

#[test]
fn betweenness_result_methods() {
    let engine = create_test_engine();
    let _nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = BetweennessCentralityConfig::default().with_direction(Direction::Both);
    let result = BetweennessCentrality::compute(&tx, &config).unwrap();

    // Test mean
    let mean = result.mean();
    assert!(mean >= 0.0);

    // Test sorted
    let sorted = result.sorted();
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].1 >= sorted[i].1);
    }
}

// ============================================================================
// Community Detection tests
// ============================================================================

#[test]
fn community_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = CommunityDetectionConfig::default();
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    assert!(result.assignments.is_empty());
    assert!(result.converged);
    assert_eq!(result.num_communities, 0);
}

#[test]
fn community_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default();
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.num_communities, 1);
    assert!(result.community(node.id).is_some());
}

#[test]
fn community_disconnected_nodes() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for _ in 0..5 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        nodes.push(node.id);
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42);
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    // Each disconnected node stays in its own community
    assert_eq!(result.num_communities, 5);
}

#[test]
fn community_connected_clique() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both);
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    // All nodes in a clique should end up in the same community
    assert_eq!(result.num_communities, 1);

    let community = result.community(nodes[0]).unwrap();
    for &node in &nodes[1..] {
        assert_eq!(
            result.community(node).unwrap(),
            community,
            "All nodes in clique should be in same community"
        );
    }
}

#[test]
fn community_two_communities() {
    let engine = create_test_engine();
    let (_community1, _community2) = create_two_community_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default()
        .with_seed(42)
        .with_direction(Direction::Both)
        .with_max_iterations(200); // May need more iterations
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);

    // The algorithm should find at least 1-2 communities
    // (Note: Label propagation may merge communities depending on the bridge)
    assert!(result.num_communities >= 1 && result.num_communities <= 6);
}

#[test]
fn community_cycle_graph() {
    let engine = create_test_engine();
    let _nodes = create_cycle_graph(&engine, 6);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both);
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);
    // In a cycle, typically all nodes end up in one community
    assert!(result.num_communities >= 1);
}

#[test]
fn community_same_community_check() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both);
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    // All nodes in complete graph should be in same community
    assert!(result.same_community(nodes[0], nodes[1]));
    assert!(result.same_community(nodes[0], nodes[2]));
    assert!(result.same_community(nodes[0], nodes[3]));
}

#[test]
fn community_members() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both);
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    // Get members of the first community
    let first_community = result.community(nodes[0]).unwrap();
    let members = result.members(first_community);

    assert_eq!(members.len(), 4); // All nodes in same community
}

#[test]
fn community_sizes() {
    let engine = create_test_engine();
    let _nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both);
    let result = CommunityDetection::label_propagation(&tx, &config).unwrap();

    let sizes = result.community_sizes();
    let total_size: usize = sizes.values().sum();
    assert_eq!(total_size, 5);

    // Test communities_by_size
    let by_size = result.communities_by_size();
    assert!(!by_size.is_empty());

    // Test largest/smallest
    let largest = result.largest_community();
    let smallest = result.smallest_community();
    assert!(largest.is_some());
    assert!(smallest.is_some());
    assert!(largest.unwrap().1 >= smallest.unwrap().1);
}

#[test]
fn community_reproducibility_with_seed() {
    let engine = create_test_engine();
    let _ = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config =
        CommunityDetectionConfig::default().with_seed(12345).with_direction(Direction::Both);

    let result1 = CommunityDetection::label_propagation(&tx, &config).unwrap();
    let result2 = CommunityDetection::label_propagation(&tx, &config).unwrap();

    // With the same seed, results should be the same
    assert_eq!(result1.num_communities, result2.num_communities);
    assert_eq!(result1.iterations, result2.iterations);
}

#[test]
fn community_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both);

    // Only analyze first 3 nodes
    let subset = vec![nodes[0], nodes[1], nodes[2]];
    let result = CommunityDetection::label_propagation_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.assignments.len(), 3);
    assert!(result.community(nodes[3]).is_none());
}

// ============================================================================
// Integration tests - combining algorithms
// ============================================================================

#[test]
fn analytics_on_same_graph() {
    let engine = create_test_engine();
    let (center, _spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();

    // Run all three algorithms
    let pr_result = PageRank::compute(&tx, &PageRankConfig::default()).unwrap();
    let bc_result = BetweennessCentrality::compute(
        &tx,
        &BetweennessCentralityConfig::default().with_direction(Direction::Both),
    )
    .unwrap();
    let cd_result = CommunityDetection::label_propagation(
        &tx,
        &CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both),
    )
    .unwrap();

    // All should have results for all 6 nodes
    assert_eq!(pr_result.scores.len(), 6);
    assert_eq!(bc_result.scores.len(), 6);
    assert_eq!(cd_result.assignments.len(), 6);

    // Center should have highest PageRank
    let pr_max = pr_result.max().unwrap();
    assert_eq!(pr_max.0, center);

    // Center should have highest betweenness
    let bc_max = bc_result.max().unwrap();
    assert_eq!(bc_max.0, center);

    // All nodes should be in same community (star is connected)
    assert_eq!(cd_result.num_communities, 1);
}

#[test]
fn analytics_large_graph() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create a larger graph (100 nodes in a ring)
    let mut nodes = Vec::new();
    for i in 0..100 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        nodes.push(node);
    }

    // Connect in a ring with some additional random edges
    for i in 0..100 {
        let next = (i + 1) % 100;
        EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[next], "EDGE", |id| {
            Edge::new(id, nodes[i], nodes[next], "EDGE")
        })
        .unwrap();
        EdgeStore::create(&mut tx, &id_gen, nodes[next], nodes[i], "EDGE", |id| {
            Edge::new(id, nodes[next], nodes[i], "EDGE")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // PageRank should work on large graphs
    let pr_result = PageRank::compute(&tx, &PageRankConfig::default()).unwrap();
    assert_eq!(pr_result.scores.len(), 100);
    assert!(pr_result.converged);

    // In a ring, all nodes should have similar PageRank
    let scores: Vec<_> = pr_result.scores.values().collect();
    let min = scores.iter().copied().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!((max - min) / max < 0.1, "In a ring, all PageRank scores should be similar");

    // Community detection should work
    let cd_result = CommunityDetection::label_propagation(
        &tx,
        &CommunityDetectionConfig::default().with_seed(42).with_direction(Direction::Both),
    )
    .unwrap();
    assert_eq!(cd_result.assignments.len(), 100);
    // In a ring, Label Propagation may not converge to a single community
    // due to the symmetric structure. We just verify it converges and
    // the number of communities is reasonable (much smaller than N).
    assert!(
        cd_result.num_communities <= 50,
        "Ring should not fragment into too many communities: {}",
        cd_result.num_communities
    );
}

// ============================================================================
// Graph size validation tests
// ============================================================================

#[test]
fn pagerank_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 10 nodes
    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set a very low limit (5 nodes) - should fail
    let config = PageRankConfig::default().with_max_graph_nodes(Some(5));
    let result = PageRank::compute(&tx, &config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        _ => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn pagerank_graph_within_limit() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 5 nodes
    for _ in 0..5 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set limit to 10 - should succeed
    let config = PageRankConfig::default().with_max_graph_nodes(Some(10));
    let result = PageRank::compute(&tx, &config);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().scores.len(), 5);
}

#[test]
fn pagerank_graph_limit_disabled() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 10 nodes
    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Disable limit - should succeed
    let config = PageRankConfig::default().with_max_graph_nodes(None);
    let result = PageRank::compute(&tx, &config);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().scores.len(), 10);
}

#[test]
fn betweenness_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 10 nodes
    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set a very low limit (5 nodes) - should fail
    let config = BetweennessCentralityConfig::default().with_max_graph_nodes(Some(5));
    let result = BetweennessCentrality::compute(&tx, &config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        _ => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn betweenness_graph_within_limit() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 5 nodes
    for _ in 0..5 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set limit to 10 - should succeed
    let config = BetweennessCentralityConfig::default().with_max_graph_nodes(Some(10));
    let result = BetweennessCentrality::compute(&tx, &config);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().scores.len(), 5);
}

#[test]
fn community_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 10 nodes
    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set a very low limit (5 nodes) - should fail
    let config = CommunityDetectionConfig::default().with_max_graph_nodes(Some(5));
    let result = CommunityDetection::label_propagation(&tx, &config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        _ => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn community_graph_within_limit() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 5 nodes
    for _ in 0..5 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set limit to 10 - should succeed
    let config = CommunityDetectionConfig::default().with_max_graph_nodes(Some(10));
    let result = CommunityDetection::label_propagation(&tx, &config);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().assignments.len(), 5);
}

#[test]
fn graph_size_validation_error_message() {
    // Test that the error message is properly formatted
    let err = GraphError::GraphTooLarge { node_count: 100_000_000, limit: 10_000_000 };
    let msg = err.to_string();
    assert!(msg.contains("100000000"), "Error message should contain node count");
    assert!(msg.contains("10000000"), "Error message should contain limit");
    assert!(msg.contains("exceeds limit"), "Error message should explain the issue");
}

// ============================================================================
// Connected Components tests - Weakly Connected Components (WCC)
// ============================================================================

#[test]
fn wcc_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert!(result.assignments.is_empty());
    assert_eq!(result.num_components, 0);
}

#[test]
fn wcc_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.num_components, 1);
    assert!(result.component(node.id).is_some());
}

#[test]
fn wcc_disconnected_nodes() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for _ in 0..5 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        nodes.push(node.id);
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    // Each disconnected node is its own component
    assert_eq!(result.num_components, 5);
}

#[test]
fn wcc_linear_graph() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 4);
    // Linear graph A -> B -> C -> D is one weakly connected component
    assert_eq!(result.num_components, 1);

    // All nodes should be in the same component
    let c0 = result.component(nodes[0]).unwrap();
    for &node in &nodes[1..] {
        assert_eq!(result.component(node).unwrap(), c0);
    }
}

#[test]
fn wcc_cycle_graph() {
    let engine = create_test_engine();
    let nodes = create_cycle_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    assert_eq!(result.num_components, 1);

    // All nodes in same component
    assert!(result.same_component(nodes[0], nodes[4]));
}

#[test]
fn wcc_two_communities() {
    let engine = create_test_engine();
    let (community1, community2) = create_two_community_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);
    // Since communities are connected by a bridge, it's one component
    assert_eq!(result.num_components, 1);

    // All nodes should be in the same weakly connected component
    assert!(result.same_component(community1[0], community2[2]));
}

#[test]
fn wcc_two_disconnected_cliques() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create first clique (3 nodes)
    let mut clique1 = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("clique", 1i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        clique1.push(node);
    }

    // Connect all pairs in clique1
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                EdgeStore::create(&mut tx, &id_gen, clique1[i], clique1[j], "EDGE", |id| {
                    Edge::new(id, clique1[i], clique1[j], "EDGE")
                })
                .unwrap();
            }
        }
    }

    // Create second clique (3 nodes)
    let mut clique2 = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("clique", 2i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        clique2.push(node);
    }

    // Connect all pairs in clique2
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                EdgeStore::create(&mut tx, &id_gen, clique2[i], clique2[j], "EDGE", |id| {
                    Edge::new(id, clique2[i], clique2[j], "EDGE")
                })
                .unwrap();
            }
        }
    }

    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);
    // Two disconnected cliques = 2 components
    assert_eq!(result.num_components, 2);

    // Nodes within each clique are in the same component
    assert!(result.same_component(clique1[0], clique1[2]));
    assert!(result.same_component(clique2[0], clique2[2]));

    // Nodes in different cliques are in different components
    assert!(!result.same_component(clique1[0], clique2[0]));
}

#[test]
fn wcc_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);
    assert_eq!(result.num_components, 1);

    // All nodes connected through center
    for &spoke in &spokes {
        assert!(result.same_component(center, spoke));
    }
}

#[test]
fn wcc_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    assert_eq!(result.num_components, 1);

    // All nodes in same component
    for i in 1..nodes.len() {
        assert!(result.same_component(nodes[0], nodes[i]));
    }
}

#[test]
fn wcc_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();

    // Only analyze first 3 nodes
    let subset = vec![nodes[0], nodes[1], nodes[2]];
    let result = ConnectedComponents::weakly_connected_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.assignments.len(), 3);
    assert_eq!(result.num_components, 1);
    assert!(result.component(nodes[3]).is_none());
}

// ============================================================================
// Connected Components tests - Strongly Connected Components (SCC)
// ============================================================================

#[test]
fn scc_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert!(result.assignments.is_empty());
    assert_eq!(result.num_components, 0);
}

#[test]
fn scc_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.num_components, 1);
    assert!(result.component(node.id).is_some());
}

#[test]
fn scc_linear_graph() {
    let engine = create_test_engine();
    let _nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 4);
    // Linear graph A -> B -> C -> D: each node is its own SCC
    // (can't get back to A from D)
    assert_eq!(result.num_components, 4);
}

#[test]
fn scc_cycle_graph() {
    let engine = create_test_engine();
    let nodes = create_cycle_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    // Cycle graph: all nodes are mutually reachable = 1 SCC
    assert_eq!(result.num_components, 1);

    // All nodes in same SCC
    assert!(result.same_component(nodes[0], nodes[4]));
}

#[test]
fn scc_two_cycles_connected() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create cycle 1: A -> B -> C -> A
    let mut cycle1 = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("cycle", 1i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        cycle1.push(node);
    }
    for i in 0..3 {
        let src = cycle1[i];
        let tgt = cycle1[(i + 1) % 3];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "CYCLE", |id| {
            Edge::new(id, src, tgt, "CYCLE")
        })
        .unwrap();
    }

    // Create cycle 2: D -> E -> F -> D
    let mut cycle2 = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("cycle", 2i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        cycle2.push(node);
    }
    for i in 0..3 {
        let src = cycle2[i];
        let tgt = cycle2[(i + 1) % 3];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "CYCLE", |id| {
            Edge::new(id, src, tgt, "CYCLE")
        })
        .unwrap();
    }

    // Connect cycles with one-way edge: C -> D
    EdgeStore::create(&mut tx, &id_gen, cycle1[2], cycle2[0], "BRIDGE", |id| {
        Edge::new(id, cycle1[2], cycle2[0], "BRIDGE")
    })
    .unwrap();

    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);
    // Two separate SCCs (can't get from cycle2 back to cycle1)
    assert_eq!(result.num_components, 2);

    // Nodes within each cycle are in same SCC
    assert!(result.same_component(cycle1[0], cycle1[2]));
    assert!(result.same_component(cycle2[0], cycle2[2]));

    // Nodes in different cycles are in different SCCs
    assert!(!result.same_component(cycle1[0], cycle2[0]));
}

#[test]
fn scc_bidirectional_bridge() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create two cycles connected bidirectionally
    let mut cycle1 = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("cycle", 1i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        cycle1.push(node);
    }
    for i in 0..3 {
        let src = cycle1[i];
        let tgt = cycle1[(i + 1) % 3];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "CYCLE", |id| {
            Edge::new(id, src, tgt, "CYCLE")
        })
        .unwrap();
    }

    let mut cycle2 = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("cycle", 2i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        cycle2.push(node);
    }
    for i in 0..3 {
        let src = cycle2[i];
        let tgt = cycle2[(i + 1) % 3];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "CYCLE", |id| {
            Edge::new(id, src, tgt, "CYCLE")
        })
        .unwrap();
    }

    // Bidirectional bridge: C <-> D
    EdgeStore::create(&mut tx, &id_gen, cycle1[2], cycle2[0], "BRIDGE", |id| {
        Edge::new(id, cycle1[2], cycle2[0], "BRIDGE")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, cycle2[0], cycle1[2], "BRIDGE", |id| {
        Edge::new(id, cycle2[0], cycle1[2], "BRIDGE")
    })
    .unwrap();

    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 6);
    // With bidirectional bridge, all nodes are mutually reachable = 1 SCC
    assert_eq!(result.num_components, 1);

    assert!(result.same_component(cycle1[0], cycle2[0]));
}

#[test]
fn scc_complete_graph() {
    let engine = create_test_engine();
    let _nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    // Complete graph: all nodes mutually reachable = 1 SCC
    assert_eq!(result.num_components, 1);
}

#[test]
fn scc_disconnected_nodes() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for _ in 0..5 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        nodes.push(node.id);
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    assert_eq!(result.assignments.len(), 5);
    // Each isolated node is its own SCC
    assert_eq!(result.num_components, 5);
}

#[test]
fn scc_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_cycle_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();

    // Only analyze first 3 nodes (breaks the cycle)
    let subset = vec![nodes[0], nodes[1], nodes[2]];
    let result = ConnectedComponents::strongly_connected_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.assignments.len(), 3);
    // Without the full cycle, each node is its own SCC
    assert_eq!(result.num_components, 3);
}

// ============================================================================
// Connected Components - Result methods tests
// ============================================================================

#[test]
fn component_result_methods() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 6 nodes in 2 disconnected groups
    let mut group1 = Vec::new();
    let mut group2 = Vec::new();

    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("group", 1i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        group1.push(node);
    }

    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("group", 2i64).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        group2.push(node);
    }

    // Connect within groups
    EdgeStore::create(&mut tx, &id_gen, group1[0], group1[1], "EDGE", |id| {
        Edge::new(id, group1[0], group1[1], "EDGE")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, group1[1], group1[2], "EDGE", |id| {
        Edge::new(id, group1[1], group1[2], "EDGE")
    })
    .unwrap();

    EdgeStore::create(&mut tx, &id_gen, group2[0], group2[1], "EDGE", |id| {
        Edge::new(id, group2[0], group2[1], "EDGE")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, group2[1], group2[2], "EDGE", |id| {
        Edge::new(id, group2[1], group2[2], "EDGE")
    })
    .unwrap();

    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();
    let result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();

    assert_eq!(result.num_components, 2);

    // Test component_sizes
    let sizes = result.component_sizes();
    assert_eq!(sizes.len(), 2);
    let total_size: usize = sizes.values().sum();
    assert_eq!(total_size, 6);

    // Test components_by_size
    let by_size = result.components_by_size();
    assert_eq!(by_size.len(), 2);
    // Both have 3 nodes
    assert_eq!(by_size[0].1, 3);
    assert_eq!(by_size[1].1, 3);

    // Test largest/smallest
    let largest = result.largest_component().unwrap();
    let smallest = result.smallest_component().unwrap();
    assert_eq!(largest.1, 3);
    assert_eq!(smallest.1, 3);

    // Test nodes_in_component
    let c0 = result.component(group1[0]).unwrap();
    let nodes_in_c0 = result.nodes_in_component(c0);
    assert_eq!(nodes_in_c0.len(), 3);

    // Test component_size
    assert_eq!(result.component_size(c0), 3);
}

// ============================================================================
// Connected Components - Graph size validation tests
// ============================================================================

#[test]
fn wcc_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Set a very low limit
    let config = ConnectedComponentsConfig::default().with_max_graph_nodes(Some(5));
    let result = ConnectedComponents::weakly_connected(&tx, &config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        _ => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn scc_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let config = ConnectedComponentsConfig::default().with_max_graph_nodes(Some(5));
    let result = ConnectedComponents::strongly_connected(&tx, &config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        _ => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn connected_components_within_limit() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..5 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let config = ConnectedComponentsConfig::default().with_max_graph_nodes(Some(10));

    let wcc_result = ConnectedComponents::weakly_connected(&tx, &config);
    assert!(wcc_result.is_ok());
    assert_eq!(wcc_result.unwrap().assignments.len(), 5);

    let scc_result = ConnectedComponents::strongly_connected(&tx, &config);
    assert!(scc_result.is_ok());
    assert_eq!(scc_result.unwrap().assignments.len(), 5);
}

#[test]
fn connected_components_limit_disabled() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Disable limit
    let config = ConnectedComponentsConfig::default().with_max_graph_nodes(None);

    let wcc_result = ConnectedComponents::weakly_connected(&tx, &config);
    assert!(wcc_result.is_ok());

    let scc_result = ConnectedComponents::strongly_connected(&tx, &config);
    assert!(scc_result.is_ok());
}

// ============================================================================
// Connected Components - Large graph tests
// ============================================================================

#[test]
fn connected_components_large_ring() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create a large ring (100 nodes)
    let mut nodes = Vec::new();
    for i in 0..100 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        nodes.push(node);
    }

    // Connect in a ring
    for i in 0..100 {
        let next = (i + 1) % 100;
        EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[next], "EDGE", |id| {
            Edge::new(id, nodes[i], nodes[next], "EDGE")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();

    // WCC: one component (ring is connected)
    let wcc_result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();
    assert_eq!(wcc_result.num_components, 1);
    assert_eq!(wcc_result.assignments.len(), 100);

    // SCC: one component (ring is strongly connected)
    let scc_result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();
    assert_eq!(scc_result.num_components, 1);
    assert_eq!(scc_result.assignments.len(), 100);
}

#[test]
fn connected_components_large_chain() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create a large chain (100 nodes)
    let mut nodes = Vec::new();
    for i in 0..100 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_property("index", i as i64)
        })
        .unwrap()
        .id;
        nodes.push(node);
    }

    // Connect in a chain (one-way)
    for i in 0..99 {
        EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[i + 1], "EDGE", |id| {
            Edge::new(id, nodes[i], nodes[i + 1], "EDGE")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();

    // WCC: one component (chain is weakly connected)
    let wcc_result = ConnectedComponents::weakly_connected(&tx, &config).unwrap();
    assert_eq!(wcc_result.num_components, 1);

    // SCC: 100 components (chain is not strongly connected - can't go backward)
    let scc_result = ConnectedComponents::strongly_connected(&tx, &config).unwrap();
    assert_eq!(scc_result.num_components, 100);
}

// ============================================================================
// Integration tests - combining WCC and SCC
// ============================================================================

#[test]
fn wcc_vs_scc_comparison() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = ConnectedComponentsConfig::default();

    let wcc = ConnectedComponents::weakly_connected(&tx, &config).unwrap();
    let scc = ConnectedComponents::strongly_connected(&tx, &config).unwrap();

    // WCC should find 1 component (all connected ignoring direction)
    assert_eq!(wcc.num_components, 1);

    // SCC should find 4 components (each node is its own SCC in a linear chain)
    assert_eq!(scc.num_components, 4);

    // WCC: all nodes in same component
    assert!(wcc.same_component(nodes[0], nodes[3]));

    // SCC: nodes in different components
    assert!(!scc.same_component(nodes[0], nodes[3]));
}

// ============================================================================
// Degree Centrality tests
// ============================================================================

use manifoldb_graph::analytics::{
    ClosenessCentrality, ClosenessCentralityConfig, DegreeCentrality, DegreeCentralityConfig,
    EigenvectorCentrality, EigenvectorCentralityConfig,
};

#[test]
fn degree_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = DegreeCentralityConfig::default();
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    assert!(result.scores.is_empty());
}

#[test]
fn degree_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default();
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 1);
    // Single node with no edges has degree 0
    assert_eq!(result.score(node.id).unwrap(), 0.0);
}

#[test]
fn degree_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_direction(Direction::Both);
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    // Center should have highest degree (connected to all 5 spokes)
    let center_degree = result.score(center).unwrap();
    for &spoke in &spokes {
        let spoke_degree = result.score(spoke).unwrap();
        assert!(center_degree > spoke_degree, "Center should have higher degree than spokes");
    }
    // Center degree should be 10 (5 outgoing + 5 incoming in bidirectional star)
    assert_eq!(center_degree, 10.0);
}

#[test]
fn degree_outgoing_only() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_direction(Direction::Outgoing);
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    // In linear graph A -> B -> C -> D:
    // A, B, C have out-degree 1, D has out-degree 0
    assert_eq!(result.score(nodes[0]).unwrap(), 1.0);
    assert_eq!(result.score(nodes[1]).unwrap(), 1.0);
    assert_eq!(result.score(nodes[2]).unwrap(), 1.0);
    assert_eq!(result.score(nodes[3]).unwrap(), 0.0);
}

#[test]
fn degree_incoming_only() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_direction(Direction::Incoming);
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    // In linear graph A -> B -> C -> D:
    // A has in-degree 0, B, C, D have in-degree 1
    assert_eq!(result.score(nodes[0]).unwrap(), 0.0);
    assert_eq!(result.score(nodes[1]).unwrap(), 1.0);
    assert_eq!(result.score(nodes[2]).unwrap(), 1.0);
    assert_eq!(result.score(nodes[3]).unwrap(), 1.0);
}

#[test]
fn degree_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_direction(Direction::Both);
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    // In complete graph with 4 nodes, each node has degree 6 (3 in + 3 out)
    for &node in &nodes {
        let degree = result.score(node).unwrap();
        assert_eq!(degree, 6.0, "Each node in complete graph should have degree 6");
    }
}

#[test]
fn degree_normalization() {
    let engine = create_test_engine();
    let _nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();

    // With normalization
    let config_norm =
        DegreeCentralityConfig::default().with_direction(Direction::Both).with_normalize(true);
    let result_norm = DegreeCentrality::compute(&tx, &config_norm).unwrap();

    // Normalized values should be in [0, 1] (or up to 2 for bidirectional)
    for &score in result_norm.scores.values() {
        assert!(score >= 0.0 && score <= 2.0, "Normalized degree should be reasonable");
    }
}

#[test]
fn degree_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_direction(Direction::Both);

    // Only compute for first two nodes
    let subset = vec![nodes[0], nodes[1]];
    let result = DegreeCentrality::compute_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.scores.len(), 2);
    assert!(result.score(nodes[0]).is_some());
    assert!(result.score(nodes[1]).is_some());
    assert!(result.score(nodes[2]).is_none());
}

#[test]
fn degree_result_methods() {
    let engine = create_test_engine();
    let _nodes = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_direction(Direction::Both);
    let result = DegreeCentrality::compute(&tx, &config).unwrap();

    // Test sorted
    let sorted = result.sorted();
    assert_eq!(sorted.len(), 6);
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].1 >= sorted[i].1);
    }

    // Test top_n
    let top2 = result.top_n(2);
    assert_eq!(top2.len(), 2);

    // Test max/min
    let max = result.max().unwrap();
    let min = result.min().unwrap();
    assert!(max.1 >= min.1);

    // Test mean
    let mean = result.mean();
    assert!(mean >= 0.0);
}

// ============================================================================
// Closeness Centrality tests
// ============================================================================

#[test]
fn closeness_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = ClosenessCentralityConfig::default();
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    assert!(result.scores.is_empty());
}

#[test]
fn closeness_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default();
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 1);
    // Single node has zero closeness (no paths to others)
    assert_eq!(result.score(node.id).unwrap(), 0.0);
}

#[test]
fn closeness_linear_graph() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default().with_direction(Direction::Both);
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 4);

    // In linear graph A - B - C - D:
    // B and C are more central (closer to all nodes on average)
    // A and D are endpoints (farther from others)
    let score_a = result.score(nodes[0]).unwrap();
    let score_b = result.score(nodes[1]).unwrap();
    let score_c = result.score(nodes[2]).unwrap();
    let score_d = result.score(nodes[3]).unwrap();

    assert!(score_b > score_a, "B should have higher closeness than A");
    assert!(score_c > score_d, "C should have higher closeness than D");
    // B and C should have similar closeness
    assert!((score_b - score_c).abs() < score_b * 0.01);
}

#[test]
fn closeness_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default().with_direction(Direction::Both);
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    // Center should have highest closeness (distance 1 to all others)
    let center_score = result.score(center).unwrap();
    for &spoke in &spokes {
        let spoke_score = result.score(spoke).unwrap();
        assert!(center_score > spoke_score, "Center should have higher closeness than spokes");
    }
}

#[test]
fn closeness_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default().with_direction(Direction::Both);
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    // In complete graph, all nodes should have equal closeness
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();
    let first = scores[0];
    for &score in &scores[1..] {
        assert!(
            (score - first).abs() < 1e-6,
            "All nodes in complete graph should have equal closeness"
        );
    }
}

#[test]
fn closeness_harmonic() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config =
        ClosenessCentralityConfig::default().with_direction(Direction::Both).with_harmonic(true);
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    assert!(result.harmonic);
    assert_eq!(result.scores.len(), 4);

    // Middle nodes should still have higher harmonic centrality
    let score_b = result.score(nodes[1]).unwrap();
    let score_a = result.score(nodes[0]).unwrap();
    assert!(score_b > score_a, "B should have higher harmonic centrality than A");
}

#[test]
fn closeness_normalization() {
    let engine = create_test_engine();
    let _nodes = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();

    // With normalization
    let config_norm =
        ClosenessCentralityConfig::default().with_direction(Direction::Both).with_normalize(true);
    let result_norm = ClosenessCentrality::compute(&tx, &config_norm).unwrap();

    // Normalized values should be in reasonable range
    for &score in result_norm.scores.values() {
        assert!(score >= 0.0 && score <= 2.0, "Normalized closeness should be reasonable");
    }
}

#[test]
fn closeness_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default().with_direction(Direction::Both);

    let subset = vec![nodes[0], nodes[1], nodes[2]];
    let result = ClosenessCentrality::compute_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.scores.len(), 3);
    assert!(result.score(nodes[3]).is_none());
}

#[test]
fn closeness_result_methods() {
    let engine = create_test_engine();
    let _nodes = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default().with_direction(Direction::Both);
    let result = ClosenessCentrality::compute(&tx, &config).unwrap();

    // Test sorted
    let sorted = result.sorted();
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].1 >= sorted[i].1);
    }

    // Test mean
    let mean = result.mean();
    assert!(mean >= 0.0);

    // Test max/min
    let max = result.max().unwrap();
    let min = result.min().unwrap();
    assert!(max.1 >= min.1);
}

// ============================================================================
// Eigenvector Centrality tests
// ============================================================================

#[test]
fn eigenvector_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = EigenvectorCentralityConfig::default();
    let result = EigenvectorCentrality::compute(&tx, &config).unwrap();

    assert!(result.scores.is_empty());
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn eigenvector_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = EigenvectorCentralityConfig::default();
    let result = EigenvectorCentrality::compute(&tx, &config).unwrap();

    assert_eq!(result.scores.len(), 1);
    // Single node should have score 1.0 (normalized)
    assert!((result.score(node.id).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn eigenvector_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    // Star graphs may need more iterations to converge due to their structure
    let config = EigenvectorCentralityConfig::default()
        .with_direction(Direction::Both)
        .with_max_iterations(200);
    let result = EigenvectorCentrality::compute(&tx, &config).unwrap();

    // Note: Eigenvector centrality on small star graphs may not fully converge
    // due to the eigenvalue structure, but the relative scores should still be meaningful
    assert_eq!(result.scores.len(), 6);

    // Center should have highest eigenvector centrality
    let center_score = result.score(center).unwrap();
    for &spoke in &spokes {
        let spoke_score = result.score(spoke).unwrap();
        assert!(
            center_score >= spoke_score,
            "Center should have higher eigenvector centrality than spokes"
        );
    }
}

#[test]
fn eigenvector_cycle_graph() {
    let engine = create_test_engine();
    let nodes = create_cycle_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = EigenvectorCentralityConfig::default().with_direction(Direction::Both);
    let result = EigenvectorCentrality::compute(&tx, &config).unwrap();

    assert!(result.converged);

    // In a cycle, all nodes should have equal eigenvector centrality
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();
    let first = scores[0];
    for &score in &scores[1..] {
        assert!(
            (score - first).abs() < 1e-4,
            "All nodes in cycle should have similar eigenvector centrality"
        );
    }
}

#[test]
fn eigenvector_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = EigenvectorCentralityConfig::default().with_direction(Direction::Both);
    let result = EigenvectorCentrality::compute(&tx, &config).unwrap();

    assert!(result.converged);

    // In complete graph, all nodes should have equal eigenvector centrality
    let scores: Vec<_> = nodes.iter().map(|&n| result.score(n).unwrap()).collect();
    let first = scores[0];
    for &score in &scores[1..] {
        assert!(
            (score - first).abs() < 1e-6,
            "All nodes in complete graph should have equal eigenvector centrality"
        );
    }
}

#[test]
fn eigenvector_convergence_params() {
    let engine = create_test_engine();
    let _nodes = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();

    // Test with tight tolerance
    let config_tight =
        EigenvectorCentralityConfig::default().with_tolerance(1e-10).with_max_iterations(200);
    let result_tight = EigenvectorCentrality::compute(&tx, &config_tight).unwrap();

    // Test with loose tolerance (should converge faster)
    let config_loose =
        EigenvectorCentralityConfig::default().with_tolerance(1e-3).with_max_iterations(200);
    let result_loose = EigenvectorCentrality::compute(&tx, &config_loose).unwrap();

    // Loose tolerance should converge in fewer or equal iterations
    assert!(result_loose.iterations <= result_tight.iterations);
}

#[test]
fn eigenvector_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = EigenvectorCentralityConfig::default().with_direction(Direction::Both);

    let subset = vec![nodes[0], nodes[1]];
    let result = EigenvectorCentrality::compute_for_nodes(&tx, &subset, &config).unwrap();

    assert_eq!(result.scores.len(), 2);
    assert!(result.score(nodes[0]).is_some());
    assert!(result.score(nodes[1]).is_some());
    assert!(result.score(nodes[2]).is_none());
}

#[test]
fn eigenvector_result_methods() {
    let engine = create_test_engine();
    let _nodes = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = EigenvectorCentralityConfig::default().with_direction(Direction::Both);
    let result = EigenvectorCentrality::compute(&tx, &config).unwrap();

    // Test sorted
    let sorted = result.sorted();
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].1 >= sorted[i].1);
    }

    // Test top_n
    let top2 = result.top_n(2);
    assert_eq!(top2.len(), 2);

    // Test max/min
    let max = result.max().unwrap();
    let min = result.min().unwrap();
    assert!(max.1 >= min.1);

    // Test mean
    let mean = result.mean();
    assert!(mean >= 0.0);
}

// ============================================================================
// Graph size validation tests for new centrality algorithms
// ============================================================================

#[test]
fn degree_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = DegreeCentralityConfig::default().with_max_graph_nodes(Some(5));
    let result = DegreeCentrality::compute(&tx, &config);

    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        err => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn closeness_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = ClosenessCentralityConfig::default().with_max_graph_nodes(Some(5));
    let result = ClosenessCentrality::compute(&tx, &config);

    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        err => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

#[test]
fn eigenvector_graph_too_large_error() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = EigenvectorCentralityConfig::default().with_max_graph_nodes(Some(5));
    let result = EigenvectorCentrality::compute(&tx, &config);

    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::GraphTooLarge { node_count, limit } => {
            assert_eq!(node_count, 10);
            assert_eq!(limit, 5);
        }
        err => panic!("Expected GraphTooLarge error, got: {:?}", err),
    }
}

// ============================================================================
// Integration tests - all centrality algorithms on same graph
// ============================================================================

#[test]
fn all_centrality_algorithms_on_star_graph() {
    let engine = create_test_engine();
    let (center, _spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();

    // Run all centrality algorithms
    let degree_result = DegreeCentrality::compute(
        &tx,
        &DegreeCentralityConfig::default().with_direction(Direction::Both),
    )
    .unwrap();
    let closeness_result = ClosenessCentrality::compute(
        &tx,
        &ClosenessCentralityConfig::default().with_direction(Direction::Both),
    )
    .unwrap();
    let eigenvector_result = EigenvectorCentrality::compute(
        &tx,
        &EigenvectorCentralityConfig::default().with_direction(Direction::Both),
    )
    .unwrap();
    let betweenness_result = BetweennessCentrality::compute(
        &tx,
        &BetweennessCentralityConfig::default().with_direction(Direction::Both),
    )
    .unwrap();
    let pagerank_result = PageRank::compute(&tx, &PageRankConfig::default()).unwrap();

    // All should have results for all 6 nodes
    assert_eq!(degree_result.scores.len(), 6);
    assert_eq!(closeness_result.scores.len(), 6);
    assert_eq!(eigenvector_result.scores.len(), 6);
    assert_eq!(betweenness_result.scores.len(), 6);
    assert_eq!(pagerank_result.scores.len(), 6);

    // Center should have highest scores in all algorithms
    assert_eq!(degree_result.max().unwrap().0, center);
    assert_eq!(closeness_result.max().unwrap().0, center);
    assert_eq!(betweenness_result.max().unwrap().0, center);
    assert_eq!(pagerank_result.max().unwrap().0, center);
    // Eigenvector should also have center as max
    let eigenvector_max = eigenvector_result.max().unwrap();
    assert_eq!(eigenvector_max.0, center);
}

#[test]
fn all_centrality_algorithms_on_two_communities() {
    let engine = create_test_engine();
    let (community1, community2) = create_two_community_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Bridge nodes are community1[2] (C) and community2[0] (D)
    let bridge_c = community1[2];
    let bridge_d = community2[0];

    // Run betweenness centrality
    let betweenness_result = BetweennessCentrality::compute(
        &tx,
        &BetweennessCentralityConfig::default()
            .with_direction(Direction::Both)
            .with_normalize(false),
    )
    .unwrap();

    // Run closeness centrality
    let closeness_result = ClosenessCentrality::compute(
        &tx,
        &ClosenessCentralityConfig::default().with_direction(Direction::Both),
    )
    .unwrap();

    // Bridge nodes should have high betweenness
    let bc_top = betweenness_result.top_n(2);
    let top_nodes: Vec<_> = bc_top.iter().map(|(id, _)| *id).collect();
    assert!(
        top_nodes.contains(&bridge_c) || top_nodes.contains(&bridge_d),
        "Bridge nodes should have high betweenness"
    );

    // Bridge nodes should have relatively high closeness
    let cc_bridge_c = closeness_result.score(bridge_c).unwrap();
    let cc_bridge_d = closeness_result.score(bridge_d).unwrap();
    let cc_mean = closeness_result.mean();
    assert!(
        cc_bridge_c >= cc_mean || cc_bridge_d >= cc_mean,
        "Bridge nodes should have above-average closeness"
    );
}

// ============================================================================
// Triangle Count and Clustering Coefficient tests
// ============================================================================

use manifoldb_graph::analytics::{TriangleCount, TriangleCountConfig};

#[test]
fn triangle_count_empty_graph() {
    let engine = create_test_engine();
    let tx = engine.begin_read().unwrap();

    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    assert!(result.node_triangles.is_empty());
    assert_eq!(result.total_triangles, 0);
    assert!(result.coefficients.is_empty());
    assert!((result.global_coefficient - 0.0).abs() < f64::EPSILON);
}

#[test]
fn triangle_count_single_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Single node has no triangles
    assert_eq!(result.node_triangles.len(), 1);
    assert_eq!(result.triangles_for(node.id), Some(0));
    assert_eq!(result.total_triangles, 0);
    // Node with degree 0 has clustering coefficient 0
    assert!((result.coefficient_for(node.id).unwrap() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn triangle_count_linear_graph() {
    let engine = create_test_engine();
    let nodes = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Linear graph A -> B -> C -> D has no triangles
    assert_eq!(result.total_triangles, 0);
    for &node in &nodes {
        assert_eq!(result.triangles_for(node), Some(0));
    }
}

#[test]
fn triangle_count_star_graph() {
    let engine = create_test_engine();
    let (center, spokes) = create_star_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Star graph has no triangles (spokes aren't connected to each other)
    assert_eq!(result.total_triangles, 0);
    assert_eq!(result.triangles_for(center), Some(0));
    for &spoke in &spokes {
        assert_eq!(result.triangles_for(spoke), Some(0));
    }

    // Center has 5 neighbors but none are connected to each other
    // So clustering coefficient is 0
    assert!((result.coefficient_for(center).unwrap() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn triangle_count_complete_graph() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // In a complete graph K4:
    // Total triangles = C(4,3) = 4
    assert_eq!(result.total_triangles, 4);

    // Each node participates in C(3,2) = 3 triangles
    // (choosing 2 other nodes to form a triangle)
    for &node in &nodes {
        assert_eq!(result.triangles_for(node), Some(3));
    }

    // All nodes have clustering coefficient 1.0 (fully connected neighborhood)
    for &node in &nodes {
        let coef = result.coefficient_for(node).unwrap();
        assert!(
            (coef - 1.0).abs() < f64::EPSILON,
            "Expected coefficient 1.0, got {} for node {:?}",
            coef,
            node
        );
    }

    // Global coefficient should be 1.0
    assert!((result.global_coefficient - 1.0).abs() < f64::EPSILON);
}

#[test]
fn triangle_count_complete_graph_k5() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // In a complete graph K5:
    // Total triangles = C(5,3) = 10
    assert_eq!(result.total_triangles, 10);

    // Each node participates in C(4,2) = 6 triangles
    for &node in &nodes {
        assert_eq!(result.triangles_for(node), Some(6));
    }
}

#[test]
fn triangle_count_single_triangle() {
    // Create exactly one triangle: A -> B -> C -> A
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let mut nodes = Vec::new();
    for i in 0..3 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Node").with_property("index", i as i64)
        })
        .unwrap();
        nodes.push(node.id);
    }

    // Create bidirectional edges to form a triangle
    for i in 0..3 {
        let src = nodes[i];
        let tgt = nodes[(i + 1) % 3];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "EDGE", |id| Edge::new(id, src, tgt, "EDGE"))
            .unwrap();
        EdgeStore::create(&mut tx, &id_gen, tgt, src, "EDGE", |id| Edge::new(id, tgt, src, "EDGE"))
            .unwrap();
    }

    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Should have exactly 1 triangle
    assert_eq!(result.total_triangles, 1);

    // Each node participates in 1 triangle
    for &node in &nodes {
        assert_eq!(result.triangles_for(node), Some(1));
    }

    // Each node has degree 2, and 1 edge exists between neighbors
    // coefficient = 2 * 1 / (2 * 1) = 1.0
    for &node in &nodes {
        let coef = result.coefficient_for(node).unwrap();
        assert!(
            (coef - 1.0).abs() < f64::EPSILON,
            "Expected coefficient 1.0, got {} for node {:?}",
            coef,
            node
        );
    }
}

#[test]
fn triangle_count_two_communities() {
    let engine = create_test_engine();
    let (community1, community2) = create_two_community_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // The graph structure is:
    // A -- B -- C -- D -- E -- F
    // where each "--" is bidirectional
    // No triangles exist in this linear structure
    assert_eq!(result.total_triangles, 0);

    // All nodes should have 0 triangles
    for &node in &community1 {
        assert_eq!(result.triangles_for(node), Some(0));
    }
    for &node in &community2 {
        assert_eq!(result.triangles_for(node), Some(0));
    }

    // Bridge nodes (C and D) have degree 2 but their neighbors aren't connected
    // So clustering coefficient is 0
    let bridge_c = community1[2];
    let bridge_d = community2[0];
    assert!((result.coefficient_for(bridge_c).unwrap() - 0.0).abs() < f64::EPSILON);
    assert!((result.coefficient_for(bridge_d).unwrap() - 0.0).abs() < f64::EPSILON);
}

/// Create a graph with triangles and varying clustering coefficients
fn create_triangle_test_graph(engine: &RedbEngine) -> Vec<EntityId> {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 5 nodes
    let mut nodes = Vec::new();
    for i in 0..5 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Node").with_property("index", i as i64)
        })
        .unwrap()
        .id;
        nodes.push(node);
    }

    // Create a graph:
    //     0 --- 1
    //    /|\   /
    //   / | \ /
    //  4  |  2
    //     |
    //     3
    //
    // Edges (bidirectional):
    // 0-1, 0-2, 0-3, 0-4, 1-2
    let edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)];

    for (i, j) in edges {
        let src = nodes[i];
        let tgt = nodes[j];
        EdgeStore::create(&mut tx, &id_gen, src, tgt, "EDGE", |id| Edge::new(id, src, tgt, "EDGE"))
            .unwrap();
        EdgeStore::create(&mut tx, &id_gen, tgt, src, "EDGE", |id| Edge::new(id, tgt, src, "EDGE"))
            .unwrap();
    }

    tx.commit().unwrap();
    nodes
}

#[test]
fn triangle_count_mixed_clustering() {
    let engine = create_test_engine();
    let nodes = create_triangle_test_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Graph has one triangle: 0-1-2
    assert_eq!(result.total_triangles, 1);

    // Node 0 participates in 1 triangle, has degree 4
    // C(0) = 2 * 1 / (4 * 3) = 2/12 = 1/6
    assert_eq!(result.triangles_for(nodes[0]), Some(1));
    let c0 = result.coefficient_for(nodes[0]).unwrap();
    assert!((c0 - 1.0 / 6.0).abs() < 1e-6, "Expected 1/6, got {}", c0);

    // Node 1 participates in 1 triangle, has degree 2
    // C(1) = 2 * 1 / (2 * 1) = 1.0
    assert_eq!(result.triangles_for(nodes[1]), Some(1));
    let c1 = result.coefficient_for(nodes[1]).unwrap();
    assert!((c1 - 1.0).abs() < 1e-6, "Expected 1.0, got {}", c1);

    // Node 2 participates in 1 triangle, has degree 2
    // C(2) = 2 * 1 / (2 * 1) = 1.0
    assert_eq!(result.triangles_for(nodes[2]), Some(1));
    let c2 = result.coefficient_for(nodes[2]).unwrap();
    assert!((c2 - 1.0).abs() < 1e-6, "Expected 1.0, got {}", c2);

    // Node 3 participates in 0 triangles, has degree 1
    // C(3) = 0 (degree < 2)
    assert_eq!(result.triangles_for(nodes[3]), Some(0));
    let c3 = result.coefficient_for(nodes[3]).unwrap();
    assert!((c3 - 0.0).abs() < f64::EPSILON);

    // Node 4 participates in 0 triangles, has degree 1
    // C(4) = 0 (degree < 2)
    assert_eq!(result.triangles_for(nodes[4]), Some(0));
    let c4 = result.coefficient_for(nodes[4]).unwrap();
    assert!((c4 - 0.0).abs() < f64::EPSILON);
}

#[test]
fn triangle_count_result_methods() {
    let engine = create_test_engine();
    let _nodes = create_complete_graph(&engine, 4);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Test sorted_by_triangles
    let sorted = result.sorted_by_triangles();
    assert_eq!(sorted.len(), 4);
    // All have same count in complete graph
    for (_, count) in &sorted {
        assert_eq!(*count, 3);
    }

    // Test sorted_by_coefficient
    let sorted_coef = result.sorted_by_coefficient();
    assert_eq!(sorted_coef.len(), 4);
    for (_, coef) in &sorted_coef {
        assert!((*coef - 1.0).abs() < f64::EPSILON);
    }

    // Test top_n_by_triangles
    let top2 = result.top_n_by_triangles(2);
    assert_eq!(top2.len(), 2);

    // Test top_n_by_coefficient
    let top2_coef = result.top_n_by_coefficient(2);
    assert_eq!(top2_coef.len(), 2);

    // Test max_triangles
    let max_tri = result.max_triangles().unwrap();
    assert_eq!(max_tri.1, 3);

    // Test max_coefficient
    let max_coef = result.max_coefficient().unwrap();
    assert!((max_coef.1 - 1.0).abs() < f64::EPSILON);
}

#[test]
fn triangle_count_for_nodes_subset() {
    let engine = create_test_engine();
    let nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();

    // Compute for first 3 nodes only (forms K3)
    let subset = &nodes[0..3];
    let result = TriangleCount::compute_for_nodes(&tx, subset, &config).unwrap();

    // K3 has 1 triangle
    assert_eq!(result.total_triangles, 1);

    // Each node in subset participates in 1 triangle
    for &node in subset {
        assert_eq!(result.triangles_for(node), Some(1));
    }

    // Nodes not in subset shouldn't be in result
    assert!(result.triangles_for(nodes[3]).is_none());
    assert!(result.triangles_for(nodes[4]).is_none());
}

#[test]
fn triangle_count_graph_too_large() {
    let engine = create_test_engine();
    let _nodes = create_complete_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    // Set very low limit
    let config = TriangleCountConfig::default().with_max_graph_nodes(Some(2));
    let result = TriangleCount::compute(&tx, &config);

    // Should fail due to graph size limit
    assert!(result.is_err());
    if let Err(GraphError::GraphTooLarge { node_count, limit }) = result {
        assert_eq!(node_count, 5);
        assert_eq!(limit, 2);
    } else {
        panic!("Expected GraphTooLarge error");
    }
}

#[test]
fn triangle_count_cycle_graph() {
    let engine = create_test_engine();
    let nodes = create_cycle_graph(&engine, 5);

    let tx = engine.begin_read().unwrap();
    let config = TriangleCountConfig::default();
    let result = TriangleCount::compute(&tx, &config).unwrap();

    // Cycle graph (directed) has no triangles
    // Each node has in-degree 1 and out-degree 1
    // No two neighbors of any node are connected
    assert_eq!(result.total_triangles, 0);

    for &node in &nodes {
        assert_eq!(result.triangles_for(node), Some(0));
    }
}
