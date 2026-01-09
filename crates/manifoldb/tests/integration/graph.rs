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
// SQL MATCH Query Tests
// ============================================================================

#[test]
fn test_sql_match_single_hop() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create persons
    let alice =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Alice");
    let bob = tx.create_entity().expect("failed").with_label("Person").with_property("name", "Bob");
    let carol =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Carol");

    tx.put_entity(&alice).expect("failed");
    tx.put_entity(&bob).expect("failed");
    tx.put_entity(&carol).expect("failed");

    // Create FOLLOWS relationships
    let e1 = tx.create_edge(alice.id, bob.id, "FOLLOWS").expect("failed");
    let e2 = tx.create_edge(alice.id, carol.id, "FOLLOWS").expect("failed");
    let e3 = tx.create_edge(bob.id, carol.id, "FOLLOWS").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");
    tx.put_edge(&e3).expect("failed");

    tx.commit().expect("failed");

    // Execute MATCH query using Database.query()
    // Test: Find all FOLLOWS relationships (3 total: Alice->Bob, Alice->Carol, Bob->Carol)
    let result = db.query("SELECT * FROM Person p MATCH (p)-[:FOLLOWS]->(f)");

    // The query should return all 3 FOLLOWS relationships
    assert!(result.is_ok(), "query should succeed: {:?}", result);
    let result = result.unwrap();

    // We should have 3 relationships
    assert_eq!(result.len(), 3, "Should find 3 FOLLOWS relationships, got {}", result.len());
}

#[test]
fn test_sql_match_with_filter() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create persons with different activity status
    let alice = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "Alice")
        .with_property("active", true);
    let bob = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "Bob")
        .with_property("active", true);
    let carol = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "Carol")
        .with_property("active", false);

    tx.put_entity(&alice).expect("failed");
    tx.put_entity(&bob).expect("failed");
    tx.put_entity(&carol).expect("failed");

    // Create FOLLOWS relationships
    let e1 = tx.create_edge(alice.id, bob.id, "FOLLOWS").expect("failed");
    let e2 = tx.create_edge(alice.id, carol.id, "FOLLOWS").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");

    tx.commit().expect("failed");

    // Execute MATCH query - this tests the basic structure exists
    let result = db.query("SELECT * FROM Person p MATCH (p)-[:FOLLOWS]->(f)");

    assert!(result.is_ok(), "query should succeed: {:?}", result);
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

// ============================================================================
// OPTIONAL MATCH Tests
// ============================================================================

/// Creates a test graph for OPTIONAL MATCH testing:
/// - 3 users: Alice (with posts), Bob (with posts), Carol (no posts)
/// - 2 posts by Alice, 1 post by Bob
fn create_optional_match_test_graph(db: &Database) -> (Vec<EntityId>, Vec<EntityId>) {
    let mut tx = db.begin().expect("failed to begin");

    // Create users
    let alice = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("name", "Alice")
        .with_property("active", true);
    let alice_id = alice.id;
    tx.put_entity(&alice).expect("failed");

    let bob = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("name", "Bob")
        .with_property("active", true);
    let bob_id = bob.id;
    tx.put_entity(&bob).expect("failed");

    let carol = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("name", "Carol")
        .with_property("active", true);
    let carol_id = carol.id;
    tx.put_entity(&carol).expect("failed");

    // Create posts
    let post1 = tx
        .create_entity()
        .expect("failed")
        .with_label("Post")
        .with_property("title", "Alice Post 1");
    let post1_id = post1.id;
    tx.put_entity(&post1).expect("failed");

    let post2 = tx
        .create_entity()
        .expect("failed")
        .with_label("Post")
        .with_property("title", "Alice Post 2");
    let post2_id = post2.id;
    tx.put_entity(&post2).expect("failed");

    let post3 =
        tx.create_entity().expect("failed").with_label("Post").with_property("title", "Bob Post 1");
    let post3_id = post3.id;
    tx.put_entity(&post3).expect("failed");

    // Create WROTE edges
    let edge = tx.create_edge(alice_id, post1_id, "WROTE").expect("failed");
    tx.put_edge(&edge).expect("failed");

    let edge = tx.create_edge(alice_id, post2_id, "WROTE").expect("failed");
    tx.put_edge(&edge).expect("failed");

    let edge = tx.create_edge(bob_id, post3_id, "WROTE").expect("failed");
    tx.put_edge(&edge).expect("failed");

    // Carol has no posts

    tx.commit().expect("failed");

    (vec![alice_id, bob_id, carol_id], vec![post1_id, post2_id, post3_id])
}

#[test]
fn test_optional_match_basic_setup() {
    // Basic test to verify the test graph is created correctly
    let db = Database::in_memory().expect("failed to create db");
    let (users, posts) = create_optional_match_test_graph(&db);

    assert_eq!(users.len(), 3);
    assert_eq!(posts.len(), 3);

    let tx = db.begin_read().expect("failed to begin");

    // Alice (users[0]) should have 2 outgoing WROTE edges
    let alice_edges = tx.get_outgoing_edges(users[0]).expect("failed");
    assert_eq!(alice_edges.len(), 2);

    // Bob (users[1]) should have 1 outgoing WROTE edge
    let bob_edges = tx.get_outgoing_edges(users[1]).expect("failed");
    assert_eq!(bob_edges.len(), 1);

    // Carol (users[2]) should have 0 outgoing edges
    let carol_edges = tx.get_outgoing_edges(users[2]).expect("failed");
    assert!(carol_edges.is_empty());
}

#[test]
fn test_optional_match_users_with_and_without_posts() {
    // This test verifies the graph structure that OPTIONAL MATCH would query:
    // "List all users with their posts (if any)"
    //
    // With OPTIONAL MATCH:
    // - Alice should appear with 2 posts
    // - Bob should appear with 1 post
    // - Carol should appear with NULL for post (no posts)
    //
    // Without OPTIONAL MATCH (regular MATCH):
    // - Carol would NOT appear in results (inner join semantics)

    let db = Database::in_memory().expect("failed to create db");
    let (users, _posts) = create_optional_match_test_graph(&db);

    let tx = db.begin_read().expect("failed to begin");

    // Count users who have at least one post (would be returned by regular MATCH)
    let users_with_posts: Vec<_> = users
        .iter()
        .filter(|&&user_id| {
            let edges = tx.get_outgoing_edges(user_id).expect("failed");
            !edges.is_empty()
        })
        .collect();

    // Only Alice and Bob have posts
    assert_eq!(users_with_posts.len(), 2);

    // All users should be returned by OPTIONAL MATCH (including Carol)
    // This is the expected behavior of OPTIONAL MATCH
    let all_users_count = users.len();
    assert_eq!(all_users_count, 3);
}

#[test]
fn test_optional_match_null_handling_concept() {
    // This test demonstrates what OPTIONAL MATCH achieves:
    // When there's no matching relationship, the optional side returns NULL
    //
    // In our case:
    // - Carol has no WROTE relationship to any Post
    // - OPTIONAL MATCH (carol)-[:WROTE]->(p:Post) would return:
    //   carol = Carol, p = NULL

    let db = Database::in_memory().expect("failed to create db");
    let (users, _posts) = create_optional_match_test_graph(&db);

    let tx = db.begin_read().expect("failed to begin");

    // Carol is the third user
    let carol_id = users[2];
    let carol_entity = tx.get_entity(carol_id).expect("failed to get entity");

    assert!(carol_entity.is_some());
    let carol = carol_entity.unwrap();
    assert_eq!(carol.get_property("name"), Some(&Value::from("Carol")));

    // Carol has no outgoing edges, so in OPTIONAL MATCH, her posts would be NULL
    let carol_posts = tx.get_outgoing_edges(carol_id).expect("failed");
    assert!(carol_posts.is_empty());

    // This is the key difference:
    // - Regular MATCH: Carol would not appear in results
    // - OPTIONAL MATCH: Carol appears with NULL for post variables
}

// ============================================================================
// Variable-Length Path Tests
// ============================================================================

/// Creates a test graph for variable-length path tests:
/// A -> B -> C -> D (linear chain)
/// With KNOWS relationship type
fn create_linear_chain_for_paths(db: &Database) -> Vec<EntityId> {
    let mut tx = db.begin().expect("failed to begin");

    let a = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "Alice")
        .with_property("depth", 0i64);
    let b = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "Bob")
        .with_property("depth", 1i64);
    let c = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "Carol")
        .with_property("depth", 2i64);
    let d = tx
        .create_entity()
        .expect("failed")
        .with_label("Person")
        .with_property("name", "David")
        .with_property("depth", 3i64);

    tx.put_entity(&a).expect("failed");
    tx.put_entity(&b).expect("failed");
    tx.put_entity(&c).expect("failed");
    tx.put_entity(&d).expect("failed");

    // A -> B -> C -> D chain
    let e1 = tx.create_edge(a.id, b.id, "KNOWS").expect("failed");
    let e2 = tx.create_edge(b.id, c.id, "KNOWS").expect("failed");
    let e3 = tx.create_edge(c.id, d.id, "KNOWS").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");
    tx.put_edge(&e3).expect("failed");

    tx.commit().expect("failed");

    vec![a.id, b.id, c.id, d.id]
}

/// Creates a friends-of-friends graph for testing:
/// Alice -> Bob, Charlie (direct friends)
/// Bob -> David, Eve (Bob's friends = Alice's friends-of-friends)
/// Charlie -> Frank (Charlie's friend = Alice's friend-of-friend)
fn create_friends_of_friends_graph(db: &Database) -> Vec<EntityId> {
    let mut tx = db.begin().expect("failed to begin");

    let alice =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Alice");
    let bob = tx.create_entity().expect("failed").with_label("Person").with_property("name", "Bob");
    let charlie =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Charlie");
    let david =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "David");
    let eve = tx.create_entity().expect("failed").with_label("Person").with_property("name", "Eve");
    let frank =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Frank");

    tx.put_entity(&alice).expect("failed");
    tx.put_entity(&bob).expect("failed");
    tx.put_entity(&charlie).expect("failed");
    tx.put_entity(&david).expect("failed");
    tx.put_entity(&eve).expect("failed");
    tx.put_entity(&frank).expect("failed");

    // Alice's direct friends
    let e1 = tx.create_edge(alice.id, bob.id, "FRIEND").expect("failed");
    let e2 = tx.create_edge(alice.id, charlie.id, "FRIEND").expect("failed");

    // Bob's friends (Alice's friends-of-friends)
    let e3 = tx.create_edge(bob.id, david.id, "FRIEND").expect("failed");
    let e4 = tx.create_edge(bob.id, eve.id, "FRIEND").expect("failed");

    // Charlie's friend
    let e5 = tx.create_edge(charlie.id, frank.id, "FRIEND").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");
    tx.put_edge(&e3).expect("failed");
    tx.put_edge(&e4).expect("failed");
    tx.put_edge(&e5).expect("failed");

    tx.commit().expect("failed");

    vec![alice.id, bob.id, charlie.id, david.id, eve.id, frank.id]
}

/// Creates a graph with a cycle for testing cycle detection:
/// A -> B -> C -> A (triangle)
fn create_cycle_graph(db: &Database) -> Vec<EntityId> {
    let mut tx = db.begin().expect("failed to begin");

    let a = tx.create_entity().expect("failed").with_label("Node").with_property("name", "A");
    let b = tx.create_entity().expect("failed").with_label("Node").with_property("name", "B");
    let c = tx.create_entity().expect("failed").with_label("Node").with_property("name", "C");

    tx.put_entity(&a).expect("failed");
    tx.put_entity(&b).expect("failed");
    tx.put_entity(&c).expect("failed");

    let e1 = tx.create_edge(a.id, b.id, "EDGE").expect("failed");
    let e2 = tx.create_edge(b.id, c.id, "EDGE").expect("failed");
    let e3 = tx.create_edge(c.id, a.id, "EDGE").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");
    tx.put_edge(&e3).expect("failed");

    tx.commit().expect("failed");

    vec![a.id, b.id, c.id]
}

#[test]
fn test_variable_length_path_single_hop() {
    // Test [*1..1] - equivalent to single hop
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // From A (depth 0), should reach only B (depth 1) with [*1..1]
    let a_edges = tx.get_outgoing_edges(nodes[0]).expect("failed");
    assert_eq!(a_edges.len(), 1);
    assert_eq!(a_edges[0].target, nodes[1]);
}

#[test]
fn test_variable_length_path_exact_depth() {
    // Test [*2] - exact depth of 2
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // From A, 2 hops should reach C
    let a_edges = tx.get_outgoing_edges(nodes[0]).expect("failed");
    assert_eq!(a_edges.len(), 1);
    let b_id = a_edges[0].target;

    let b_edges = tx.get_outgoing_edges(b_id).expect("failed");
    assert_eq!(b_edges.len(), 1);
    assert_eq!(b_edges[0].target, nodes[2]); // C
}

#[test]
fn test_variable_length_path_range() {
    // Test [*1..3] - depths 1 to 3
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // BFS traversal from A should reach B, C, D (depths 1, 2, 3)
    let mut reachable = HashSet::new();
    let mut queue = vec![nodes[0]];
    let mut visited = HashSet::new();
    let mut depth_map: std::collections::HashMap<EntityId, usize> =
        std::collections::HashMap::new();

    visited.insert(nodes[0]);
    depth_map.insert(nodes[0], 0);

    while !queue.is_empty() {
        let mut next_queue = Vec::new();
        for node in queue {
            let edges = tx.get_outgoing_edges(node).expect("failed");
            for edge in edges {
                if !visited.contains(&edge.target) {
                    visited.insert(edge.target);
                    let new_depth = depth_map[&node] + 1;
                    depth_map.insert(edge.target, new_depth);

                    if new_depth >= 1 && new_depth <= 3 {
                        reachable.insert(edge.target);
                    }

                    if new_depth < 3 {
                        next_queue.push(edge.target);
                    }
                }
            }
        }
        queue = next_queue;
    }

    // Should have B, C, D
    assert_eq!(reachable.len(), 3);
    assert!(reachable.contains(&nodes[1])); // B
    assert!(reachable.contains(&nodes[2])); // C
    assert!(reachable.contains(&nodes[3])); // D
}

#[test]
fn test_friends_of_friends() {
    // Test [*2..2] for friends-of-friends (exactly 2 hops away)
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_friends_of_friends_graph(&db);

    let tx = db.begin_read().expect("failed to begin");
    let alice = nodes[0];
    let david = nodes[3];
    let eve = nodes[4];
    let frank = nodes[5];

    // BFS from Alice to find friends-of-friends (depth 2)
    let mut friends_of_friends = HashSet::new();

    // First hop: get Alice's direct friends
    let alice_friends = tx.get_outgoing_edges(alice).expect("failed");
    assert_eq!(alice_friends.len(), 2); // Bob and Charlie

    // Second hop: get friends of Alice's friends
    for friend in &alice_friends {
        let fof = tx.get_outgoing_edges(friend.target).expect("failed");
        for f in fof {
            friends_of_friends.insert(f.target);
        }
    }

    // Alice's friends-of-friends should be David, Eve, Frank
    assert_eq!(friends_of_friends.len(), 3);
    assert!(friends_of_friends.contains(&david));
    assert!(friends_of_friends.contains(&eve));
    assert!(friends_of_friends.contains(&frank));
}

#[test]
fn test_variable_length_path_min_only() {
    // Test [*2..] - minimum depth 2, no maximum
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // From A with min depth 2, should reach C and D (not B)
    let mut reachable = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = vec![(nodes[0], 0usize)];
    visited.insert(nodes[0]);

    while let Some((node, depth)) = queue.pop() {
        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target);
                let new_depth = depth + 1;

                // Only add if depth >= 2
                if new_depth >= 2 {
                    reachable.insert(edge.target);
                }

                queue.push((edge.target, new_depth));
            }
        }
    }

    // Should reach C (depth 2) and D (depth 3), but not B (depth 1)
    assert_eq!(reachable.len(), 2);
    assert!(!reachable.contains(&nodes[1])); // B - excluded (depth 1)
    assert!(reachable.contains(&nodes[2])); // C
    assert!(reachable.contains(&nodes[3])); // D
}

#[test]
fn test_variable_length_path_max_only() {
    // Test [*..2] - maximum depth 2, minimum 1
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // From A with max depth 2, should reach B and C (not D)
    let mut reachable = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = vec![(nodes[0], 0usize)];
    visited.insert(nodes[0]);

    while let Some((node, depth)) = queue.pop() {
        if depth >= 2 {
            continue; // Don't expand beyond max depth
        }

        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target);
                let new_depth = depth + 1;

                // Only add if depth >= 1 and <= 2
                if new_depth >= 1 && new_depth <= 2 {
                    reachable.insert(edge.target);
                }

                queue.push((edge.target, new_depth));
            }
        }
    }

    // Should reach B (depth 1) and C (depth 2), but not D (depth 3)
    assert_eq!(reachable.len(), 2);
    assert!(reachable.contains(&nodes[1])); // B
    assert!(reachable.contains(&nodes[2])); // C
    assert!(!reachable.contains(&nodes[3])); // D - excluded (depth 3)
}

#[test]
fn test_variable_length_path_unbounded() {
    // Test [*] - any depth (typically 1..)
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // From A with [*], should reach all nodes B, C, D
    let mut reachable = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = vec![nodes[0]];
    visited.insert(nodes[0]);

    while let Some(node) = queue.pop() {
        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target);
                reachable.insert(edge.target);
                queue.push(edge.target);
            }
        }
    }

    // Should reach all nodes
    assert_eq!(reachable.len(), 3);
    assert!(reachable.contains(&nodes[1])); // B
    assert!(reachable.contains(&nodes[2])); // C
    assert!(reachable.contains(&nodes[3])); // D
}

#[test]
fn test_variable_length_path_cycle_detection() {
    // Test that cycles don't cause infinite loops
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_cycle_graph(&db);

    let tx = db.begin_read().expect("failed to begin");

    // BFS from A with [*] should visit each node exactly once
    let mut visited = HashSet::new();
    let mut queue = vec![nodes[0]];
    visited.insert(nodes[0]);

    while let Some(node) = queue.pop() {
        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target);
                queue.push(edge.target);
            }
        }
    }

    // Should visit all 3 nodes exactly once (A, B, C)
    assert_eq!(visited.len(), 3);
    assert!(visited.contains(&nodes[0])); // A
    assert!(visited.contains(&nodes[1])); // B
    assert!(visited.contains(&nodes[2])); // C
}

#[test]
fn test_variable_length_path_edge_type_filter() {
    // Test [:TYPE*] - variable length with edge type filter
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create graph with mixed edge types
    let a = tx.create_entity().expect("failed").with_label("Person").with_property("name", "A");
    let b = tx.create_entity().expect("failed").with_label("Person").with_property("name", "B");
    let c = tx.create_entity().expect("failed").with_label("Person").with_property("name", "C");

    tx.put_entity(&a).expect("failed");
    tx.put_entity(&b).expect("failed");
    tx.put_entity(&c).expect("failed");

    // A -[FRIEND]-> B -[WORKS_WITH]-> C
    let e1 = tx.create_edge(a.id, b.id, "FRIEND").expect("failed");
    let e2 = tx.create_edge(b.id, c.id, "WORKS_WITH").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");

    tx.commit().expect("failed");

    // With [:FRIEND*], from A should only reach B (not C since B->C is WORKS_WITH)
    let tx = db.begin_read().expect("failed to begin");

    // Filter by FRIEND edge type
    let a_friends = tx.get_outgoing_edges(a.id).expect("failed");
    let friend_edges: Vec<_> =
        a_friends.iter().filter(|e| e.edge_type.as_str() == "FRIEND").collect();

    assert_eq!(friend_edges.len(), 1);
    assert_eq!(friend_edges[0].target, b.id);

    // B's FRIEND edges (should be empty since B->C is WORKS_WITH)
    let b_edges = tx.get_outgoing_edges(b.id).expect("failed");
    let b_friend_edges: Vec<_> =
        b_edges.iter().filter(|e| e.edge_type.as_str() == "FRIEND").collect();

    assert!(b_friend_edges.is_empty());
}

#[test]
fn test_variable_length_path_direction_outgoing() {
    // Test outgoing direction: (a)-[:KNOWS*1..2]->(x)
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // Outgoing from B should reach C and D
    let b_outgoing = tx.get_outgoing_edges(nodes[1]).expect("failed");
    assert_eq!(b_outgoing.len(), 1);
    assert_eq!(b_outgoing[0].target, nodes[2]); // C

    let c_outgoing = tx.get_outgoing_edges(nodes[2]).expect("failed");
    assert_eq!(c_outgoing.len(), 1);
    assert_eq!(c_outgoing[0].target, nodes[3]); // D
}

#[test]
fn test_variable_length_path_direction_incoming() {
    // Test incoming direction: (a)<-[:KNOWS*1..2]-(x)
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // Incoming to C should reach B and A
    let c_incoming = tx.get_incoming_edges(nodes[2]).expect("failed");
    assert_eq!(c_incoming.len(), 1);
    assert_eq!(c_incoming[0].source, nodes[1]); // B

    let b_incoming = tx.get_incoming_edges(nodes[1]).expect("failed");
    assert_eq!(b_incoming.len(), 1);
    assert_eq!(b_incoming[0].source, nodes[0]); // A
}

#[test]
fn test_variable_length_path_direction_both() {
    // Test both directions: (a)-[:KNOWS*1..2]-(x)
    let db = Database::in_memory().expect("failed to create db");
    let nodes = create_linear_chain_for_paths(&db);

    let tx = db.begin_read().expect("failed to begin");

    // From B with both directions should reach A (incoming) and C (outgoing)
    let b_outgoing = tx.get_outgoing_edges(nodes[1]).expect("failed");
    let b_incoming = tx.get_incoming_edges(nodes[1]).expect("failed");

    // Combine both directions
    let mut neighbors = HashSet::new();
    for edge in b_outgoing {
        neighbors.insert(edge.target);
    }
    for edge in b_incoming {
        neighbors.insert(edge.source);
    }

    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&nodes[0])); // A (incoming)
    assert!(neighbors.contains(&nodes[2])); // C (outgoing)
}
