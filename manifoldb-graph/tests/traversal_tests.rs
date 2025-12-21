//! Integration tests for graph traversal operators.
//!
//! These tests verify the correctness of Expand, ExpandAll, ShortestPath,
//! PathPattern, and TraversalIterator on various graph topologies.

use manifoldb_core::{Edge, EdgeType, Entity, EntityId};
use manifoldb_graph::store::{EdgeStore, IdGenerator, NodeStore};
use manifoldb_graph::traversal::{
    AllShortestPaths, Direction, Expand, ExpandAll, PathPattern, PathStep, PatternBuilder,
    ShortestPath, TraversalConfig, TraversalFilter, TraversalIterator,
};
use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{StorageEngine, Transaction};

fn create_test_engine() -> RedbEngine {
    RedbEngine::in_memory().expect("Failed to create in-memory engine")
}

/// Create a simple linear graph: A -> B -> C -> D
fn create_linear_graph(engine: &RedbEngine) -> (EntityId, EntityId, EntityId, EntityId) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Node")).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Node")).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Node")).unwrap();
    let d = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Node")).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "NEXT", |id| Edge::new(id, a.id, b.id, "NEXT"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, b.id, c.id, "NEXT", |id| Edge::new(id, b.id, c.id, "NEXT"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, d.id, "NEXT", |id| Edge::new(id, c.id, d.id, "NEXT"))
        .unwrap();

    tx.commit().unwrap();
    (a.id, b.id, c.id, d.id)
}

/// Create a star graph: center node with 5 neighbors
fn create_star_graph(engine: &RedbEngine) -> (EntityId, Vec<EntityId>) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let center =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Center")).unwrap();

    let mut leaves = Vec::new();
    for i in 0..5 {
        let leaf = NodeStore::create(&mut tx, &id_gen, |id| {
            Entity::new(id).with_label("Leaf").with_property("index", i as i64)
        })
        .unwrap();
        EdgeStore::create(&mut tx, &id_gen, center.id, leaf.id, "CONNECTS", |id| {
            Edge::new(id, center.id, leaf.id, "CONNECTS")
        })
        .unwrap();
        leaves.push(leaf.id);
    }

    tx.commit().unwrap();
    (center.id, leaves)
}

/// Create a cyclic graph: A -> B -> C -> A
fn create_cyclic_graph(engine: &RedbEngine) -> (EntityId, EntityId, EntityId) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "EDGE", |id| Edge::new(id, a.id, b.id, "EDGE"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, b.id, c.id, "EDGE", |id| Edge::new(id, b.id, c.id, "EDGE"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, a.id, "EDGE", |id| Edge::new(id, c.id, a.id, "EDGE"))
        .unwrap();

    tx.commit().unwrap();
    (a.id, b.id, c.id)
}

/// Create a graph with multiple edge types:
/// A -[FRIEND]-> B -[WORKS_AT]-> C
/// A -[FOLLOWS]-> B
fn create_multi_type_graph(engine: &RedbEngine) -> (EntityId, EntityId, EntityId) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "FRIEND", |id| {
        Edge::new(id, a.id, b.id, "FRIEND")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "FOLLOWS", |id| {
        Edge::new(id, a.id, b.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, b.id, c.id, "WORKS_AT", |id| {
        Edge::new(id, b.id, c.id, "WORKS_AT")
    })
    .unwrap();

    tx.commit().unwrap();
    (a.id, b.id, c.id)
}

/// Create a dense graph with bidirectional edges
fn create_bidirectional_graph(engine: &RedbEngine) -> (EntityId, EntityId, EntityId) {
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // A <-> B
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "KNOWS", |id| {
        Edge::new(id, a.id, b.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "KNOWS", |id| {
        Edge::new(id, b.id, a.id, "KNOWS")
    })
    .unwrap();

    // B <-> C
    EdgeStore::create(&mut tx, &id_gen, b.id, c.id, "KNOWS", |id| {
        Edge::new(id, b.id, c.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, b.id, "KNOWS", |id| {
        Edge::new(id, c.id, b.id, "KNOWS")
    })
    .unwrap();

    tx.commit().unwrap();
    (a.id, b.id, c.id)
}

// ============================================================================
// Expand tests - Single-hop traversal
// ============================================================================

#[test]
fn expand_neighbors_outgoing() {
    let engine = create_test_engine();
    let (center, leaves) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let neighbors = Expand::neighbors(&tx, center, Direction::Outgoing).unwrap();

    assert_eq!(neighbors.len(), 5);
    for result in &neighbors {
        assert!(leaves.contains(&result.node));
        assert_eq!(result.direction, Direction::Outgoing);
    }
}

#[test]
fn expand_neighbors_incoming() {
    let engine = create_test_engine();
    let (center, leaves) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    // Leaves HAVE incoming edges from center (center -> leaf)
    let neighbors = Expand::neighbors(&tx, leaves[0], Direction::Incoming).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].node, center);
    assert_eq!(neighbors[0].direction, Direction::Incoming);

    // Center has no incoming edges (only outgoing to leaves)
    let incoming = Expand::neighbors(&tx, center, Direction::Incoming).unwrap();
    assert!(incoming.is_empty());
}

#[test]
fn expand_neighbors_both_directions() {
    let engine = create_test_engine();
    let (a, b, _c) = create_bidirectional_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let neighbors = Expand::neighbors(&tx, a, Direction::Both).unwrap();

    // A should see B both as outgoing and incoming
    assert_eq!(neighbors.len(), 2);
    for result in &neighbors {
        assert_eq!(result.node, b);
    }
}

#[test]
fn expand_neighbors_by_type() {
    let engine = create_test_engine();
    let (a, b, _c) = create_multi_type_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Only FRIEND edges
    let friends =
        Expand::neighbors_by_type(&tx, a, Direction::Outgoing, &EdgeType::new("FRIEND")).unwrap();
    assert_eq!(friends.len(), 1);
    assert_eq!(friends[0].node, b);

    // Only FOLLOWS edges
    let follows =
        Expand::neighbors_by_type(&tx, a, Direction::Outgoing, &EdgeType::new("FOLLOWS")).unwrap();
    assert_eq!(follows.len(), 1);
    assert_eq!(follows[0].node, b);
}

#[test]
fn expand_neighbor_ids() {
    let engine = create_test_engine();
    let (center, leaves) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let neighbor_ids = Expand::neighbor_ids(&tx, center, Direction::Outgoing).unwrap();

    assert_eq!(neighbor_ids.len(), 5);
    for id in neighbor_ids {
        assert!(leaves.contains(&id));
    }
}

#[test]
fn expand_neighbors_with_filter() {
    let engine = create_test_engine();
    let (center, leaves) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Limit to 3 results
    let filter = TraversalFilter::new().with_limit(3);
    let neighbors = Expand::neighbors_filtered(&tx, center, Direction::Outgoing, &filter).unwrap();
    assert_eq!(neighbors.len(), 3);

    // Exclude first two leaves
    let filter = TraversalFilter::new().exclude_node(leaves[0]).exclude_node(leaves[1]);
    let neighbors = Expand::neighbors_filtered(&tx, center, Direction::Outgoing, &filter).unwrap();
    assert_eq!(neighbors.len(), 3);
    assert!(!neighbors.iter().any(|n| n.node == leaves[0]));
    assert!(!neighbors.iter().any(|n| n.node == leaves[1]));
}

#[test]
fn expand_from_multiple_nodes() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let results = Expand::expand_all(&tx, &[a, b], Direction::Outgoing).unwrap();

    // A -> B, B -> C
    assert_eq!(results.len(), 2);

    let a_neighbors: Vec<_> = results.iter().filter(|(src, _)| *src == a).collect();
    assert_eq!(a_neighbors.len(), 1);
    assert_eq!(a_neighbors[0].1.node, b);

    let b_neighbors: Vec<_> = results.iter().filter(|(src, _)| *src == b).collect();
    assert_eq!(b_neighbors.len(), 1);
    assert_eq!(b_neighbors[0].1.node, c);
}

// ============================================================================
// ExpandAll tests - Multi-hop traversal
// ============================================================================

#[test]
fn expand_all_depth_1() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(a, Direction::Outgoing).with_max_depth(1).execute(&tx).unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, b);
    assert_eq!(nodes[0].depth, 1);
}

#[test]
fn expand_all_depth_2() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(a, Direction::Outgoing).with_max_depth(2).execute(&tx).unwrap();

    assert_eq!(nodes.len(), 2);
    let ids: Vec<_> = nodes.iter().map(|n| n.id).collect();
    assert!(ids.contains(&b));
    assert!(ids.contains(&c));
}

#[test]
fn expand_all_unlimited_depth() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(a, Direction::Outgoing).execute(&tx).unwrap();

    assert_eq!(nodes.len(), 3);
    let ids: Vec<_> = nodes.iter().map(|n| n.id).collect();
    assert!(ids.contains(&b));
    assert!(ids.contains(&c));
    assert!(ids.contains(&d));
}

#[test]
fn expand_all_min_depth() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(a, Direction::Outgoing).with_min_depth(2).execute(&tx).unwrap();

    // Only nodes at depth >= 2 (C and D)
    assert_eq!(nodes.len(), 2);
    let ids: Vec<_> = nodes.iter().map(|n| n.id).collect();
    assert!(ids.contains(&c));
    assert!(ids.contains(&d));
    assert!(!ids.contains(&b));
}

#[test]
fn expand_all_depth_range() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(a, Direction::Outgoing).with_depth_range(2, 2).execute(&tx).unwrap();

    // Only nodes at exactly depth 2 (C)
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, c);
}

#[test]
fn expand_all_with_limit() {
    let engine = create_test_engine();
    let (center, _) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(center, Direction::Outgoing).with_limit(3).execute(&tx).unwrap();

    assert_eq!(nodes.len(), 3);
}

#[test]
fn expand_all_with_edge_type() {
    let engine = create_test_engine();
    let (a, b, c) = create_multi_type_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Follow FRIEND -> WORKS_AT chain
    let nodes =
        ExpandAll::new(a, Direction::Outgoing).with_edge_type("FRIEND").execute(&tx).unwrap();

    // Only B (FRIEND), not C (WORKS_AT is different type)
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, b);
}

#[test]
fn expand_all_handles_cycles() {
    let engine = create_test_engine();
    let (a, b, c) = create_cyclic_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let nodes = ExpandAll::new(a, Direction::Outgoing).execute(&tx).unwrap();

    // Should visit each node exactly once despite cycle
    assert_eq!(nodes.len(), 2);
    let ids: Vec<_> = nodes.iter().map(|n| n.id).collect();
    assert!(ids.contains(&b));
    assert!(ids.contains(&c));
}

#[test]
fn expand_all_collect_node_ids() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let ids =
        ExpandAll::new(a, Direction::Outgoing).with_max_depth(2).collect_node_ids(&tx).unwrap();

    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&b));
    assert!(ids.contains(&c));
}

#[test]
fn expand_all_count() {
    let engine = create_test_engine();
    let (center, _) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let count = ExpandAll::new(center, Direction::Outgoing).count(&tx).unwrap();

    assert_eq!(count, 5);
}

// ============================================================================
// ShortestPath tests - BFS path finding
// ============================================================================

#[test]
fn shortest_path_same_node() {
    let engine = create_test_engine();
    let (a, _, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let path = ShortestPath::new(a, a, Direction::Outgoing).find(&tx).unwrap();

    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.length, 0);
    assert_eq!(path.nodes.len(), 1);
    assert_eq!(path.nodes[0], a);
}

#[test]
fn shortest_path_direct_neighbor() {
    let engine = create_test_engine();
    let (a, b, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let path = ShortestPath::new(a, b, Direction::Outgoing).find(&tx).unwrap();

    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.length, 1);
    assert_eq!(path.source(), a);
    assert_eq!(path.target(), b);
}

#[test]
fn shortest_path_multi_hop() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let path = ShortestPath::new(a, d, Direction::Outgoing).find(&tx).unwrap();

    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.length, 3);
    assert_eq!(path.nodes, vec![a, b, c, d]);
}

#[test]
fn shortest_path_no_path() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    // D -> A doesn't exist (only outgoing direction)
    let path = ShortestPath::new(d, a, Direction::Outgoing).find(&tx).unwrap();

    assert!(path.is_none());
}

#[test]
fn shortest_path_with_max_depth() {
    let engine = create_test_engine();
    let (a, _b, _c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Max depth 2 shouldn't reach D (which is 3 hops away)
    let path = ShortestPath::new(a, d, Direction::Outgoing).with_max_depth(2).find(&tx).unwrap();

    assert!(path.is_none());

    // Max depth 3 should work
    let path = ShortestPath::new(a, d, Direction::Outgoing).with_max_depth(3).find(&tx).unwrap();

    assert!(path.is_some());
}

#[test]
fn shortest_path_both_directions() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // D -> A should work in Both direction
    let path = ShortestPath::new(d, a, Direction::Both).find(&tx).unwrap();

    assert!(path.is_some());
    assert_eq!(path.unwrap().length, 3);
}

#[test]
fn shortest_path_with_edge_type() {
    let engine = create_test_engine();
    let (a, b, c) = create_multi_type_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Path using FRIEND edge
    let path =
        ShortestPath::new(a, b, Direction::Outgoing).with_edge_type("FRIEND").find(&tx).unwrap();

    assert!(path.is_some());

    // Path A -> C requires FRIEND then WORKS_AT, but filter only allows FRIEND
    let path =
        ShortestPath::new(a, c, Direction::Outgoing).with_edge_type("FRIEND").find(&tx).unwrap();

    assert!(path.is_none());
}

#[test]
fn shortest_path_exists() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    assert!(ShortestPath::new(a, d, Direction::Outgoing).exists(&tx).unwrap());
    assert!(!ShortestPath::new(d, a, Direction::Outgoing).exists(&tx).unwrap());
}

#[test]
fn shortest_path_distance() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    assert_eq!(ShortestPath::new(a, a, Direction::Outgoing).distance(&tx).unwrap(), Some(0));
    assert_eq!(ShortestPath::new(a, b, Direction::Outgoing).distance(&tx).unwrap(), Some(1));
    assert_eq!(ShortestPath::new(a, d, Direction::Outgoing).distance(&tx).unwrap(), Some(3));
    assert_eq!(ShortestPath::new(d, a, Direction::Outgoing).distance(&tx).unwrap(), None);
}

#[test]
fn shortest_path_convenience_method() {
    let engine = create_test_engine();
    let (a, _, _, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let path = ShortestPath::find_path(&tx, a, d, Direction::Outgoing).unwrap();

    assert!(path.is_some());
    assert_eq!(path.unwrap().length, 3);
}

// ============================================================================
// AllShortestPaths tests
// ============================================================================

#[test]
fn all_shortest_paths_single() {
    let engine = create_test_engine();
    let (a, b, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let paths = AllShortestPaths::new(a, b, Direction::Outgoing).find(&tx).unwrap();

    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].length, 1);
}

#[test]
fn all_shortest_paths_same_node() {
    let engine = create_test_engine();
    let (a, _, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let paths = AllShortestPaths::new(a, a, Direction::Outgoing).find(&tx).unwrap();

    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].length, 0);
}

#[test]
fn all_shortest_paths_no_path() {
    let engine = create_test_engine();
    let (a, _, _, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let paths = AllShortestPaths::new(d, a, Direction::Outgoing).find(&tx).unwrap();

    assert!(paths.is_empty());
}

// ============================================================================
// PathPattern tests
// ============================================================================

#[test]
fn path_pattern_empty() {
    let engine = create_test_engine();
    let (a, _, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new();
    let matches = pattern.find_from(&tx, a).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].nodes.len(), 1);
    assert_eq!(matches[0].source(), a);
}

#[test]
fn path_pattern_single_step() {
    let engine = create_test_engine();
    let (a, b, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new().outgoing("NEXT");
    let matches = pattern.find_from(&tx, a).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].target(), b);
}

#[test]
fn path_pattern_multi_step() {
    let engine = create_test_engine();
    let (a, b, c, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new().outgoing("NEXT").outgoing("NEXT");
    let matches = pattern.find_from(&tx, a).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].nodes, vec![a, b, c]);
}

#[test]
fn path_pattern_with_limit() {
    let engine = create_test_engine();
    let (center, _) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new().add_step(PathStep::any(Direction::Outgoing)).with_limit(3);
    let matches = pattern.find_from(&tx, center).unwrap();

    assert_eq!(matches.len(), 3);
}

#[test]
fn path_pattern_find_between() {
    let engine = create_test_engine();
    let (a, _, c, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new().outgoing("NEXT").outgoing("NEXT");
    let matches = pattern.find_between(&tx, a, c).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].target(), c);
}

#[test]
fn path_pattern_matches() {
    let engine = create_test_engine();
    let (a, b, c, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new().outgoing("NEXT");

    assert!(pattern.clone().matches(&tx, a).unwrap());
    assert!(pattern.clone().matches(&tx, b).unwrap());
    assert!(pattern.clone().matches(&tx, c).unwrap());
}

#[test]
fn path_pattern_variable_length() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PathPattern::new().add_step(PathStep::outgoing("NEXT").variable_length(1, 3));
    let matches = pattern.find_from(&tx, a).unwrap();

    // Should match paths of length 1, 2, and 3
    assert!(matches.len() >= 3);

    let targets: Vec<_> = matches.iter().map(|m| m.target()).collect();
    assert!(targets.contains(&b)); // 1 hop
    assert!(targets.contains(&c)); // 2 hops
    assert!(targets.contains(&d)); // 3 hops
}

#[test]
fn path_pattern_builder() {
    let engine = create_test_engine();
    let (a, b, c) = create_multi_type_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern = PatternBuilder::new().out("FRIEND").out("WORKS_AT").build();

    let matches = pattern.find_from(&tx, a).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].target(), c);
}

#[test]
fn path_pattern_handles_cycles() {
    let engine = create_test_engine();
    let (a, b, c) = create_cyclic_graph(&engine);

    let tx = engine.begin_read().unwrap();
    // Without allowing cycles, should not revisit nodes
    let pattern = PathPattern::new().outgoing("EDGE").outgoing("EDGE");
    let matches = pattern.find_from(&tx, a).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].nodes, vec![a, b, c]);
}

#[test]
fn path_pattern_with_cycles_allowed() {
    let engine = create_test_engine();
    let (a, b, c) = create_cyclic_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let pattern =
        PathPattern::new().outgoing("EDGE").outgoing("EDGE").outgoing("EDGE").allow_cycles();
    let matches = pattern.find_from(&tx, a).unwrap();

    // Should complete the full cycle
    assert!(!matches.is_empty());
    assert!(matches.iter().any(|m| m.target() == a));
}

// ============================================================================
// TraversalIterator tests
// ============================================================================

#[test]
fn traversal_iterator_basic() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TraversalConfig::new(Direction::Outgoing).with_max_depth(2);
    let iter = TraversalIterator::new(&tx, a, config);
    let nodes = iter.collect_all().unwrap();

    assert_eq!(nodes.len(), 2);
}

#[test]
fn traversal_iterator_with_limit() {
    let engine = create_test_engine();
    let (center, _) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TraversalConfig::new(Direction::Outgoing).with_limit(3);
    let iter = TraversalIterator::new(&tx, center, config);
    let nodes = iter.collect_all().unwrap();

    assert_eq!(nodes.len(), 3);
}

#[test]
fn traversal_iterator_include_start() {
    let engine = create_test_engine();
    let (a, b, _, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TraversalConfig::new(Direction::Outgoing).include_start().with_max_depth(1);
    let iter = TraversalIterator::new(&tx, a, config);
    let nodes = iter.collect_all().unwrap();

    assert_eq!(nodes.len(), 2);
    assert!(nodes.iter().any(|n| n.id == a && n.depth == 0));
    assert!(nodes.iter().any(|n| n.id == b && n.depth == 1));
}

#[test]
fn traversal_iterator_collect_ids() {
    let engine = create_test_engine();
    let (a, b, c, _) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TraversalConfig::new(Direction::Outgoing).with_max_depth(2);
    let iter = TraversalIterator::new(&tx, a, config);
    let ids = iter.collect_ids().unwrap();

    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&b));
    assert!(ids.contains(&c));
}

#[test]
fn traversal_iterator_count() {
    let engine = create_test_engine();
    let (center, _) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TraversalConfig::new(Direction::Outgoing);
    let iter = TraversalIterator::new(&tx, center, config);
    let count = iter.count_all().unwrap();

    assert_eq!(count, 5);
}

#[test]
fn traversal_iterator_take() {
    let engine = create_test_engine();
    let (center, _) = create_star_graph(&engine);

    let tx = engine.begin_read().unwrap();
    let config = TraversalConfig::new(Direction::Outgoing);
    let iter = TraversalIterator::new(&tx, center, config);
    let nodes = iter.take(2).unwrap();

    assert_eq!(nodes.len(), 2);
}

// ============================================================================
// Large graph tests
// ============================================================================

#[test]
fn traversal_large_graph() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create a chain of 100 nodes
    let mut nodes = Vec::new();
    for _ in 0..100 {
        let node =
            NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Node")).unwrap();
        nodes.push(node.id);
    }

    // Connect them in a chain
    for i in 0..99 {
        EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[i + 1], "NEXT", |id| {
            Edge::new(id, nodes[i], nodes[i + 1], "NEXT")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Test BFS traversal
    let all = ExpandAll::new(nodes[0], Direction::Outgoing).execute(&tx).unwrap();
    assert_eq!(all.len(), 99); // All nodes except start

    // Test shortest path
    let path = ShortestPath::new(nodes[0], nodes[99], Direction::Outgoing).find(&tx).unwrap();
    assert!(path.is_some());
    assert_eq!(path.unwrap().length, 99);

    // Test with limit
    let limited =
        ExpandAll::new(nodes[0], Direction::Outgoing).with_limit(10).execute(&tx).unwrap();
    assert_eq!(limited.len(), 10);
}

#[test]
fn traversal_dense_graph() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();

    // Create 10 nodes with edges between many pairs
    let mut nodes = Vec::new();
    for _ in 0..10 {
        let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        nodes.push(node.id);
    }

    // Connect each node to the next 3 nodes (wrapping around)
    for i in 0..10 {
        for j in 1..=3 {
            let target = (i + j) % 10;
            EdgeStore::create(&mut tx, &id_gen, nodes[i], nodes[target], "LINK", |id| {
                Edge::new(id, nodes[i], nodes[target], "LINK")
            })
            .unwrap();
        }
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Each node should have 3 outgoing neighbors
    let neighbors = Expand::neighbors(&tx, nodes[0], Direction::Outgoing).unwrap();
    assert_eq!(neighbors.len(), 3);

    // BFS should visit all nodes
    let all = ExpandAll::new(nodes[0], Direction::Outgoing).execute(&tx).unwrap();
    assert_eq!(all.len(), 9); // All except start
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn expand_empty_graph_node() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let mut tx = engine.begin_write().unwrap();
    let lonely = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let neighbors = Expand::neighbors(&tx, lonely.id, Direction::Both).unwrap();
    assert!(neighbors.is_empty());

    let all = ExpandAll::new(lonely.id, Direction::Both).execute(&tx).unwrap();
    assert!(all.is_empty());
}

#[test]
fn shortest_path_with_excluded_nodes() {
    let engine = create_test_engine();
    let (a, b, c, d) = create_linear_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Exclude B, so path A -> D should fail
    let path = ShortestPath::new(a, d, Direction::Outgoing).exclude_nodes([b]).find(&tx).unwrap();

    assert!(path.is_none());
}

#[test]
fn traversal_filter_combination() {
    let engine = create_test_engine();
    let (a, b, c) = create_multi_type_graph(&engine);

    let tx = engine.begin_read().unwrap();

    // Filter by edge type AND limit
    let filter =
        TraversalFilter::new().with_edge_type("FRIEND").with_edge_type("FOLLOWS").with_limit(1);

    let neighbors = Expand::neighbors_filtered(&tx, a, Direction::Outgoing, &filter).unwrap();
    assert_eq!(neighbors.len(), 1);
}
