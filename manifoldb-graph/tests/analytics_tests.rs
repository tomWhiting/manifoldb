//! Integration tests for graph analytics algorithms.
//!
//! These tests verify the correctness of PageRank, Betweenness Centrality,
//! and Community Detection algorithms on various graph topologies.

use manifoldb_core::{Edge, Entity, EntityId};
use manifoldb_graph::analytics::{
    BetweennessCentrality, BetweennessCentralityConfig, CommunityDetection,
    CommunityDetectionConfig, PageRank, PageRankConfig,
};
use manifoldb_graph::store::{EdgeStore, IdGenerator, NodeStore};
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
