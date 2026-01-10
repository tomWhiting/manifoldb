//! Procedure execution integration tests.
//!
//! Tests CALL/YIELD statement execution for graph algorithms.

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple graph for testing algorithms.
/// Returns (node_ids, edge_ids).
fn create_test_graph(db: &Database) -> (Vec<EntityId>, Vec<manifoldb::EdgeId>) {
    let mut node_ids = Vec::new();
    let mut edge_ids = Vec::new();

    let mut tx = db.begin().expect("failed to begin");

    // Create nodes: A -> B -> C -> D
    //               |         |
    //               +----E----+
    for name in ["A", "B", "C", "D", "E"] {
        let entity = tx
            .create_entity()
            .expect("failed to create")
            .with_label("Node")
            .with_property("name", name);
        node_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put");
    }

    // Create edges: A->B, B->C, C->D, A->E, E->D
    let edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)];
    for (src, tgt) in edges {
        let edge = tx
            .create_edge(node_ids[src], node_ids[tgt], "CONNECTS")
            .expect("failed to create edge");
        edge_ids.push(edge.id);
        tx.put_edge(&edge).expect("failed to put edge");
    }

    tx.commit().expect("failed to commit");
    (node_ids, edge_ids)
}

// ============================================================================
// PageRank Tests
// ============================================================================

#[test]
fn test_pagerank_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute PageRank with default parameters
    let result = db.query("CALL algo.pageRank() YIELD nodeId, score").expect("query failed");

    // Should return one row per node
    assert_eq!(result.len(), node_ids.len());

    // All scores should be positive
    let score_idx = result.column_index("score").expect("score column not found");
    for row in result.rows() {
        let score = row.get(score_idx).expect("score value missing");
        match score {
            Value::Float(s) => assert!(*s > 0.0, "score should be positive"),
            _ => panic!("score should be a float"),
        }
    }
}

#[test]
fn test_pagerank_with_parameters() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute PageRank with custom parameters
    let result = db.query("CALL algo.pageRank(0.9, 50) YIELD nodeId, score").expect("query failed");

    // Should return one row per node
    assert_eq!(result.len(), node_ids.len());
}

// ============================================================================
// BFS Tests - works because it traverses from a specific node
// ============================================================================

#[test]
fn test_bfs_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute BFS from first node
    let start_id = node_ids[0].as_u64() as i64;
    let query = format!("CALL algo.bfs({}) YIELD node, depth, path", start_id);
    let result = db.query(&query).expect("query failed");

    // Should find reachable nodes
    assert!(!result.is_empty());

    // Check that depths are non-negative
    let depth_idx = result.column_index("depth").expect("depth column not found");
    for row in result.rows() {
        let depth = row.get(depth_idx).expect("depth value missing");
        match depth {
            Value::Int(d) => assert!(*d >= 0, "depth should be non-negative"),
            _ => panic!("depth should be an integer"),
        }
    }
}

// ============================================================================
// DFS Tests - works because it traverses from a specific node
// ============================================================================

#[test]
fn test_dfs_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute DFS from first node
    let start_id = node_ids[0].as_u64() as i64;
    let query = format!("CALL algo.dfs({}) YIELD node, depth, path", start_id);
    let result = db.query(&query).expect("query failed");

    // Should find reachable nodes
    assert!(!result.is_empty());
}

// ============================================================================
// Shortest Path Tests - works because it uses specific source/target nodes
// ============================================================================

#[test]
fn test_shortest_path_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Find shortest path from A to D
    let source_id = node_ids[0].as_u64() as i64;
    let target_id = node_ids[3].as_u64() as i64;
    let query = format!("CALL algo.shortestPath({}, {}) YIELD path, cost", source_id, target_id);
    let result = db.query(&query).expect("query failed");

    // Should find a path (there are two paths: A->B->C->D and A->E->D)
    assert!(!result.is_empty());
}

#[test]
fn test_shortest_path_no_path() {
    let db = Database::in_memory().expect("failed to create db");

    // Create two disconnected nodes
    let mut tx = db.begin().expect("failed to begin");
    let node1 = tx.create_entity().expect("failed").with_label("Node");
    let node2 = tx.create_entity().expect("failed").with_label("Node");
    tx.put_entity(&node1).expect("failed to put");
    tx.put_entity(&node2).expect("failed to put");
    tx.commit().expect("failed to commit");

    // Try to find path between disconnected nodes
    let source_id = node1.id.as_u64() as i64;
    let target_id = node2.id.as_u64() as i64;
    let query = format!("CALL algo.shortestPath({}, {}) YIELD path, cost", source_id, target_id);
    let result = db.query(&query).expect("query failed");

    // Should return empty result (no path exists)
    assert!(result.is_empty());
}

// ============================================================================
// Connected Components Tests
// ============================================================================

#[test]
fn test_connected_components_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute connected components
    let result = db
        .query("CALL algo.connectedComponents() YIELD nodeId, componentId")
        .expect("query failed");

    // Should return one row per node
    assert_eq!(result.len(), node_ids.len());

    // All nodes should be in the same component (graph is connected)
    let component_idx = result.column_index("componentId").expect("componentId column not found");
    let mut components = std::collections::HashSet::new();
    for row in result.rows() {
        if let Some(Value::Int(c)) = row.get(component_idx) {
            components.insert(*c);
        }
    }
    assert_eq!(components.len(), 1, "all nodes should be in the same component");
}

// ============================================================================
// Louvain Community Detection Tests
// ============================================================================

#[test]
fn test_louvain_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute Louvain algorithm
    let result = db.query("CALL algo.louvain() YIELD nodeId, communityId").expect("query failed");

    // Should return one row per node
    assert_eq!(result.len(), node_ids.len());
}

// ============================================================================
// Degree Centrality Tests
// ============================================================================

#[test]
fn test_degree_centrality_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute degree centrality
    let result = db
        .query("CALL algo.degreeCentrality() YIELD nodeId, inDegree, outDegree, totalDegree")
        .expect("query failed");

    // Should return one row per node
    assert_eq!(result.len(), node_ids.len());

    // Check that degrees are non-negative
    let degree_idx = result.column_index("totalDegree").expect("totalDegree column not found");
    for row in result.rows() {
        let degree = row.get(degree_idx).expect("degree value missing");
        match degree {
            Value::Int(d) => assert!(*d >= 0, "degree should be non-negative"),
            _ => panic!("degree should be an integer"),
        }
    }
}

// ============================================================================
// Jaccard Similarity Tests - works because it uses specific node IDs
// ============================================================================

#[test]
fn test_jaccard_similarity_execution() {
    let db = Database::in_memory().expect("failed to create db");
    let (node_ids, _) = create_test_graph(&db);

    // Execute Jaccard similarity between two nodes
    let node1_id = node_ids[0].as_u64() as i64;
    let node2_id = node_ids[1].as_u64() as i64;
    let query =
        format!("CALL algo.jaccard({}, {}) YIELD node1, node2, similarity", node1_id, node2_id);
    let result = db.query(&query).expect("query failed");

    // Should return exactly one row
    assert_eq!(result.len(), 1);

    // Similarity should be between 0 and 1
    let sim_idx = result.column_index("similarity").expect("similarity column not found");
    let sim = result.rows()[0].get(sim_idx).expect("similarity value missing");
    match sim {
        Value::Float(s) => {
            assert!(*s >= 0.0 && *s <= 1.0, "similarity should be between 0 and 1");
        }
        _ => panic!("similarity should be a float"),
    }
}

// ============================================================================
// Unknown Procedure Error Tests
// ============================================================================

#[test]
fn test_unknown_procedure_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to call a non-existent procedure
    let result = db.query("CALL algo.nonexistent() YIELD x");

    // Should return an error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Unknown procedure") || err.to_string().contains("unknown"),
        "error should mention unknown procedure"
    );
}
