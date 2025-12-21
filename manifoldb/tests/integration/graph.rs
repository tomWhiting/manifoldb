//! Graph traversal integration tests.
//!
//! Tests graph operations on different topologies:
//! - Linear chains
//! - Trees
//! - Dense graphs
//! - Bipartite graphs

use std::collections::HashSet;

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a linear chain: 1 -> 2 -> 3 -> ... -> n
fn create_linear_chain(db: &Database, n: usize) -> Vec<EntityId> {
    let mut ids = Vec::with_capacity(n);

    let mut tx = db.begin().expect("failed to begin");
    for i in 0..n {
        let entity = tx
            .create_entity()
            .expect("failed to create")
            .with_label("ChainNode")
            .with_property("position", i as i64);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put");
    }

    // Create edges between consecutive nodes
    for i in 0..(n - 1) {
        let edge = tx.create_edge(ids[i], ids[i + 1], "NEXT").expect("failed to create edge");
        tx.put_edge(&edge).expect("failed to put edge");
    }

    tx.commit().expect("failed to commit");
    ids
}

/// Create a binary tree: root with branching factor 2
fn create_binary_tree(db: &Database, depth: usize) -> Vec<EntityId> {
    let node_count = (1 << (depth + 1)) - 1; // 2^(depth+1) - 1
    let mut ids = Vec::with_capacity(node_count);

    let mut tx = db.begin().expect("failed to begin");

    // Create all nodes
    for i in 0..node_count {
        let entity = tx
            .create_entity()
            .expect("failed to create")
            .with_label("TreeNode")
            .with_property("index", i as i64)
            .with_property("depth", (64 - (i + 1).leading_zeros() - 1) as i64);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put");
    }

    // Create edges (parent -> children)
    for i in 0..node_count {
        let left_child = 2 * i + 1;
        let right_child = 2 * i + 2;

        if left_child < node_count {
            let edge =
                tx.create_edge(ids[i], ids[left_child], "LEFT_CHILD").expect("failed to create");
            tx.put_edge(&edge).expect("failed to put");
        }
        if right_child < node_count {
            let edge =
                tx.create_edge(ids[i], ids[right_child], "RIGHT_CHILD").expect("failed to create");
            tx.put_edge(&edge).expect("failed to put");
        }
    }

    tx.commit().expect("failed to commit");
    ids
}

/// Create a dense graph (k-regular-ish graph)
fn create_dense_graph(db: &Database, n: usize, k: usize) -> Vec<EntityId> {
    let mut ids = Vec::with_capacity(n);

    let mut tx = db.begin().expect("failed to begin");

    // Create nodes
    for i in 0..n {
        let entity = tx
            .create_entity()
            .expect("failed to create")
            .with_label("DenseNode")
            .with_property("index", i as i64);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put");
    }

    // Create k edges from each node
    for i in 0..n {
        for j in 1..=k {
            let target = (i + j) % n;
            if target != i {
                let edge =
                    tx.create_edge(ids[i], ids[target], "CONNECTS").expect("failed to create");
                tx.put_edge(&edge).expect("failed to put");
            }
        }
    }

    tx.commit().expect("failed to commit");
    ids
}

/// Create a bipartite graph with two sets
fn create_bipartite_graph(
    db: &Database,
    set_a_size: usize,
    set_b_size: usize,
) -> (Vec<EntityId>, Vec<EntityId>) {
    let mut set_a = Vec::with_capacity(set_a_size);
    let mut set_b = Vec::with_capacity(set_b_size);

    let mut tx = db.begin().expect("failed to begin");

    // Create set A nodes
    for i in 0..set_a_size {
        let entity =
            tx.create_entity().expect("failed").with_label("SetA").with_property("index", i as i64);
        set_a.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    // Create set B nodes
    for i in 0..set_b_size {
        let entity =
            tx.create_entity().expect("failed").with_label("SetB").with_property("index", i as i64);
        set_b.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    // Connect each node in A to all nodes in B
    for &a in &set_a {
        for &b in &set_b {
            let edge = tx.create_edge(a, b, "LINKS").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }
    }

    tx.commit().expect("failed");
    (set_a, set_b)
}

// ============================================================================
// Linear Chain Tests
// ============================================================================

#[test]
fn test_linear_chain_traversal_small() {
    let db = Database::in_memory().expect("failed to create db");
    let chain = create_linear_chain(&db, 10);

    let tx = db.begin_read().expect("failed to begin");

    // Each node (except last) should have exactly 1 outgoing edge
    for i in 0..(chain.len() - 1) {
        let edges = tx.get_outgoing_edges(chain[i]).expect("failed");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, chain[i + 1]);
    }

    // Last node should have no outgoing edges
    let edges = tx.get_outgoing_edges(chain[chain.len() - 1]).expect("failed");
    assert!(edges.is_empty());

    // Each node (except first) should have exactly 1 incoming edge
    for i in 1..chain.len() {
        let edges = tx.get_incoming_edges(chain[i]).expect("failed");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, chain[i - 1]);
    }

    // First node should have no incoming edges
    let edges = tx.get_incoming_edges(chain[0]).expect("failed");
    assert!(edges.is_empty());
}

#[test]
fn test_linear_chain_traversal_medium() {
    let db = Database::in_memory().expect("failed to create db");
    let chain = create_linear_chain(&db, 1000);

    let tx = db.begin_read().expect("failed to begin");

    // Traverse from start to end by following outgoing edges
    let mut current = chain[0];
    let mut count = 0;

    loop {
        count += 1;
        let edges = tx.get_outgoing_edges(current).expect("failed");
        if edges.is_empty() {
            break;
        }
        current = edges[0].target;
    }

    assert_eq!(count, chain.len());
    assert_eq!(current, chain[chain.len() - 1]);
}

// ============================================================================
// Binary Tree Tests
// ============================================================================

#[test]
fn test_binary_tree_structure() {
    let db = Database::in_memory().expect("failed to create db");
    let depth = 4; // 31 nodes
    let tree = create_binary_tree(&db, depth);

    let tx = db.begin_read().expect("failed to begin");

    // Root should have 2 children
    let root_edges = tx.get_outgoing_edges(tree[0]).expect("failed");
    assert_eq!(root_edges.len(), 2);

    // Internal nodes should have 2 children
    for i in 0..((1 << depth) - 1) {
        let edges = tx.get_outgoing_edges(tree[i]).expect("failed");
        assert_eq!(edges.len(), 2, "node {i} should have 2 children");
    }

    // Leaf nodes should have no children
    for i in ((1 << depth) - 1)..tree.len() {
        let edges = tx.get_outgoing_edges(tree[i]).expect("failed");
        assert!(edges.is_empty(), "leaf node {i} should have no children");
    }

    // Root should have no parents
    let root_parents = tx.get_incoming_edges(tree[0]).expect("failed");
    assert!(root_parents.is_empty());

    // Non-root nodes should have exactly 1 parent
    for i in 1..tree.len() {
        let edges = tx.get_incoming_edges(tree[i]).expect("failed");
        assert_eq!(edges.len(), 1, "node {i} should have 1 parent");
    }
}

#[test]
fn test_tree_level_by_level_traversal() {
    let db = Database::in_memory().expect("failed to create db");
    let depth = 3; // 15 nodes
    let tree = create_binary_tree(&db, depth);

    let tx = db.begin_read().expect("failed to begin");

    // BFS traversal
    let mut visited = HashSet::new();
    let mut queue = vec![tree[0]];
    let mut levels: Vec<Vec<EntityId>> = Vec::new();

    while !queue.is_empty() {
        let mut next_level = Vec::new();
        levels.push(queue.clone());

        for node in queue {
            if visited.insert(node) {
                let edges = tx.get_outgoing_edges(node).expect("failed");
                for edge in edges {
                    next_level.push(edge.target);
                }
            }
        }

        queue = next_level;
    }

    // Should have depth + 1 levels
    assert_eq!(levels.len(), depth + 1);

    // Each level should have 2^level nodes
    for (level, nodes) in levels.iter().enumerate() {
        assert_eq!(nodes.len(), 1 << level, "level {level} should have {} nodes", 1 << level);
    }
}

// ============================================================================
// Dense Graph Tests
// ============================================================================

#[test]
fn test_dense_graph_degree() {
    let db = Database::in_memory().expect("failed to create db");
    let n = 50;
    let k = 5;
    let nodes = create_dense_graph(&db, n, k);

    let tx = db.begin_read().expect("failed to begin");

    // Each node should have exactly k outgoing edges
    for &node in &nodes {
        let edges = tx.get_outgoing_edges(node).expect("failed");
        assert_eq!(edges.len(), k, "each node should have {k} outgoing edges");
    }
}

#[test]
fn test_dense_graph_reachability() {
    let db = Database::in_memory().expect("failed to create db");
    let n = 100;
    let k = 3;
    let nodes = create_dense_graph(&db, n, k);

    let tx = db.begin_read().expect("failed to begin");

    // From any node, we should be able to reach all nodes
    // (since we create a k-connected circular graph)
    let mut reachable = HashSet::new();
    let mut queue = vec![nodes[0]];

    while !queue.is_empty() {
        let current = queue.pop().expect("queue non-empty");
        if !reachable.insert(current) {
            continue;
        }

        let edges = tx.get_outgoing_edges(current).expect("failed");
        for edge in edges {
            if !reachable.contains(&edge.target) {
                queue.push(edge.target);
            }
        }
    }

    assert_eq!(reachable.len(), n, "all nodes should be reachable");
}

// ============================================================================
// Bipartite Graph Tests
// ============================================================================

#[test]
fn test_bipartite_graph_structure() {
    let db = Database::in_memory().expect("failed to create db");
    let (set_a, set_b) = create_bipartite_graph(&db, 10, 20);

    let tx = db.begin_read().expect("failed to begin");

    // Each node in A should connect to all nodes in B
    for &a in &set_a {
        let edges = tx.get_outgoing_edges(a).expect("failed");
        assert_eq!(edges.len(), set_b.len());

        let targets: HashSet<_> = edges.iter().map(|e| e.target).collect();
        for &b in &set_b {
            assert!(targets.contains(&b));
        }
    }

    // Each node in B should receive edges from all nodes in A
    for &b in &set_b {
        let edges = tx.get_incoming_edges(b).expect("failed");
        assert_eq!(edges.len(), set_a.len());

        let sources: HashSet<_> = edges.iter().map(|e| e.source).collect();
        for &a in &set_a {
            assert!(sources.contains(&a));
        }
    }
}

// ============================================================================
// Edge Type Filtering Tests
// ============================================================================

#[test]
fn test_multiple_edge_types() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create a person with multiple relationship types
    let alice =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Alice");
    let bob = tx.create_entity().expect("failed").with_label("Person").with_property("name", "Bob");
    let acme =
        tx.create_entity().expect("failed").with_label("Company").with_property("name", "Acme");

    tx.put_entity(&alice).expect("failed");
    tx.put_entity(&bob).expect("failed");
    tx.put_entity(&acme).expect("failed");

    // Different relationship types
    let e1 = tx.create_edge(alice.id, bob.id, "KNOWS").expect("failed");
    let e2 = tx.create_edge(alice.id, bob.id, "FRIENDS_WITH").expect("failed");
    let e3 = tx.create_edge(alice.id, acme.id, "WORKS_AT").expect("failed");
    let e4 = tx.create_edge(bob.id, acme.id, "WORKS_AT").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");
    tx.put_edge(&e3).expect("failed");
    tx.put_edge(&e4).expect("failed");

    tx.commit().expect("failed");

    // Query edges
    let tx = db.begin_read().expect("failed to begin");

    let alice_edges = tx.get_outgoing_edges(alice.id).expect("failed");
    assert_eq!(alice_edges.len(), 3);

    // Count by type
    let knows_count = alice_edges.iter().filter(|e| e.edge_type.as_str() == "KNOWS").count();
    let friends_count =
        alice_edges.iter().filter(|e| e.edge_type.as_str() == "FRIENDS_WITH").count();
    let works_count = alice_edges.iter().filter(|e| e.edge_type.as_str() == "WORKS_AT").count();

    assert_eq!(knows_count, 1);
    assert_eq!(friends_count, 1);
    assert_eq!(works_count, 1);
}

// ============================================================================
// Multi-hop Traversal Tests
// ============================================================================

#[test]
fn test_two_hop_neighbors() {
    let db = Database::in_memory().expect("failed to create db");
    let chain = create_linear_chain(&db, 10);

    let tx = db.begin_read().expect("failed to begin");

    // From node 0, 2-hop should reach node 2
    let one_hop = tx.get_outgoing_edges(chain[0]).expect("failed");
    assert_eq!(one_hop.len(), 1);

    let two_hop = tx.get_outgoing_edges(one_hop[0].target).expect("failed");
    assert_eq!(two_hop.len(), 1);
    assert_eq!(two_hop[0].target, chain[2]);
}

#[test]
fn test_n_hop_traversal() {
    let db = Database::in_memory().expect("failed to create db");
    let n = 20;
    let chain = create_linear_chain(&db, n);

    let tx = db.begin_read().expect("failed to begin");

    // Traverse n-1 hops from the start
    let mut current = chain[0];
    for i in 1..n {
        let edges = tx.get_outgoing_edges(current).expect("failed");
        assert_eq!(edges.len(), 1);
        current = edges[0].target;
        assert_eq!(current, chain[i]);
    }
}

// ============================================================================
// Scale Tests
// ============================================================================

#[test]
fn test_large_fan_out() {
    let db = Database::in_memory().expect("failed to create db");
    let fan_out = 1000;

    let mut tx = db.begin().expect("failed to begin");

    let hub = tx.create_entity().expect("failed").with_label("Hub");
    tx.put_entity(&hub).expect("failed");

    let mut spoke_ids = Vec::with_capacity(fan_out);
    for i in 0..fan_out {
        let spoke = tx
            .create_entity()
            .expect("failed")
            .with_label("Spoke")
            .with_property("index", i as i64);
        spoke_ids.push(spoke.id);
        tx.put_entity(&spoke).expect("failed");

        let edge = tx.create_edge(hub.id, spoke.id, "CONNECTS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Verify hub has 1000 outgoing edges
    let tx = db.begin_read().expect("failed to begin");
    let edges = tx.get_outgoing_edges(hub.id).expect("failed");
    assert_eq!(edges.len(), fan_out);

    // Verify each spoke has 1 incoming edge from hub
    for spoke_id in spoke_ids {
        let edges = tx.get_incoming_edges(spoke_id).expect("failed");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, hub.id);
    }
}

#[test]
fn test_large_fan_in() {
    let db = Database::in_memory().expect("failed to create db");
    let fan_in = 1000;

    let mut tx = db.begin().expect("failed to begin");

    let sink = tx.create_entity().expect("failed").with_label("Sink");
    tx.put_entity(&sink).expect("failed");

    let mut source_ids = Vec::with_capacity(fan_in);
    for i in 0..fan_in {
        let source = tx
            .create_entity()
            .expect("failed")
            .with_label("Source")
            .with_property("index", i as i64);
        source_ids.push(source.id);
        tx.put_entity(&source).expect("failed");

        let edge = tx.create_edge(source.id, sink.id, "POINTS_TO").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Verify sink has 1000 incoming edges
    let tx = db.begin_read().expect("failed to begin");
    let edges = tx.get_incoming_edges(sink.id).expect("failed");
    assert_eq!(edges.len(), fan_in);

    // Verify each source has 1 outgoing edge to sink
    for source_id in source_ids {
        let edges = tx.get_outgoing_edges(source_id).expect("failed");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, sink.id);
    }
}

// ============================================================================
// Edge Properties in Traversal
// ============================================================================

#[test]
fn test_weighted_edges() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create a weighted graph
    let nodes: Vec<_> = (0..5)
        .map(|i| {
            let e = tx
                .create_entity()
                .expect("failed")
                .with_label("Node")
                .with_property("id", i as i64);
            tx.put_entity(&e).expect("failed");
            e.id
        })
        .collect();

    // Create edges with weights
    let weights = [(0, 1, 1.0), (0, 2, 4.0), (1, 2, 2.0), (1, 3, 5.0), (2, 3, 1.0), (3, 4, 3.0)];

    for (src, dst, weight) in weights {
        let edge = tx
            .create_edge(nodes[src], nodes[dst], "WEIGHTED")
            .expect("failed")
            .with_property("weight", weight);
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Verify weights
    let tx = db.begin_read().expect("failed to begin");

    let edges_from_0 = tx.get_outgoing_edges(nodes[0]).expect("failed");
    assert_eq!(edges_from_0.len(), 2);

    for edge in edges_from_0 {
        let weight = edge.get_property("weight");
        assert!(weight.is_some());
        if let Some(Value::Float(w)) = weight {
            // Weight should be either 1.0 or 4.0
            assert!((w - 1.0).abs() < 0.001 || (w - 4.0).abs() < 0.001);
        }
    }
}
