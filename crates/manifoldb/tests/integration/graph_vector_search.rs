//! Integration tests for graph-constrained vector search.
//!
//! Tests the unified search API with traversal constraints, verifying that
//! vector search results are properly filtered to only include entities
//! reachable via graph traversal.

use manifoldb::collection::{DistanceMetric, PointStruct};
use manifoldb::{Database, EntityId};
use serde_json::json;

// ============================================================================
// Graph-Constrained Vector Search Tests
// ============================================================================

/// Test basic graph-constrained vector search.
///
/// Creates a simple graph:
/// - repo_a -> file_a1, file_a2, file_a3
/// - repo_b -> file_b1, file_b2
///
/// Then searches for vectors within repo_a only.
#[test]
fn test_graph_constrained_search_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a collection for symbols with embeddings
    db.create_collection("symbols")
        .expect("builder")
        .with_dense_vector("embedding", 4, DistanceMetric::DotProduct)
        .build()
        .expect("create collection");

    // Create graph structure: repos contain files
    let mut tx = db.begin().expect("begin");

    let repo_a = tx
        .create_entity()
        .expect("create")
        .with_label("Repository")
        .with_property("name", "repo-a");
    let repo_b = tx
        .create_entity()
        .expect("create")
        .with_label("Repository")
        .with_property("name", "repo-b");

    tx.put_entity(&repo_a).expect("put");
    tx.put_entity(&repo_b).expect("put");

    // Create files (symbols) with embeddings
    let file_a1 =
        tx.create_entity().expect("create").with_label("Symbol").with_property("name", "class_a1");
    let file_a2 =
        tx.create_entity().expect("create").with_label("Symbol").with_property("name", "class_a2");
    let file_a3 =
        tx.create_entity().expect("create").with_label("Symbol").with_property("name", "class_a3");
    let file_b1 =
        tx.create_entity().expect("create").with_label("Symbol").with_property("name", "class_b1");
    let file_b2 =
        tx.create_entity().expect("create").with_label("Symbol").with_property("name", "class_b2");

    tx.put_entity(&file_a1).expect("put");
    tx.put_entity(&file_a2).expect("put");
    tx.put_entity(&file_a3).expect("put");
    tx.put_entity(&file_b1).expect("put");
    tx.put_entity(&file_b2).expect("put");

    // Create CONTAINS edges: repo -> files
    let e1 = tx.create_edge(repo_a.id, file_a1.id, "CONTAINS").expect("edge");
    let e2 = tx.create_edge(repo_a.id, file_a2.id, "CONTAINS").expect("edge");
    let e3 = tx.create_edge(repo_a.id, file_a3.id, "CONTAINS").expect("edge");
    let e4 = tx.create_edge(repo_b.id, file_b1.id, "CONTAINS").expect("edge");
    let e5 = tx.create_edge(repo_b.id, file_b2.id, "CONTAINS").expect("edge");

    tx.put_edge(&e1).expect("put");
    tx.put_edge(&e2).expect("put");
    tx.put_edge(&e3).expect("put");
    tx.put_edge(&e4).expect("put");
    tx.put_edge(&e5).expect("put");

    tx.commit().expect("commit");

    // Add embeddings to the collection
    // Note: The symbol embeddings are stored in the collection, not the graph
    let collection = db.collection("symbols").expect("collection");

    // file_a1: [1.0, 0.0, 0.0, 0.0] - will be closest to query
    // file_a2: [0.7, 0.7, 0.0, 0.0]
    // file_a3: [0.5, 0.5, 0.5, 0.0]
    // file_b1: [0.9, 0.3, 0.0, 0.0] - closer than file_a2, but in repo_b
    // file_b2: [0.0, 1.0, 0.0, 0.0]

    collection
        .upsert_point(
            PointStruct::new(file_a1.id.as_u64())
                .with_payload(json!({"name": "class_a1"}))
                .with_vector("embedding", vec![1.0, 0.0, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(file_a2.id.as_u64())
                .with_payload(json!({"name": "class_a2"}))
                .with_vector("embedding", vec![0.7, 0.7, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(file_a3.id.as_u64())
                .with_payload(json!({"name": "class_a3"}))
                .with_vector("embedding", vec![0.5, 0.5, 0.5, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(file_b1.id.as_u64())
                .with_payload(json!({"name": "class_b1"}))
                .with_vector("embedding", vec![0.9, 0.3, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(file_b2.id.as_u64())
                .with_payload(json!({"name": "class_b2"}))
                .with_vector("embedding", vec![0.0, 1.0, 0.0, 0.0]),
        )
        .expect("insert");

    // Search without traversal constraint - should return file_b1 as second result
    let unconstrained = db
        .search("symbols", "embedding")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .limit(5)
        .execute()
        .expect("search");

    assert_eq!(unconstrained.len(), 5);
    // file_a1 should be first (exact match)
    assert_eq!(unconstrained[0].entity.id, file_a1.id);
    // Without constraint, file_b1 (0.9, 0.3) is closer than file_a2 (0.7, 0.7)
    // to query (1.0, 0.0) with dot product

    // Search WITH traversal constraint - only return files in repo_a
    let constrained = db
        .search("symbols", "embedding")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .within_traversal(repo_a.id, |p| p.edge_out("CONTAINS"))
        .limit(5)
        .execute()
        .expect("search");

    // Should only return 3 results (files in repo_a)
    assert_eq!(constrained.len(), 3);

    // All results should be from repo_a
    let result_ids: Vec<EntityId> = constrained.iter().map(|r| r.entity.id).collect();
    assert!(result_ids.contains(&file_a1.id), "should contain file_a1");
    assert!(result_ids.contains(&file_a2.id), "should contain file_a2");
    assert!(result_ids.contains(&file_a3.id), "should contain file_a3");
    assert!(!result_ids.contains(&file_b1.id), "should NOT contain file_b1");
    assert!(!result_ids.contains(&file_b2.id), "should NOT contain file_b2");

    // Results should be ordered by similarity
    assert_eq!(constrained[0].entity.id, file_a1.id, "file_a1 should be first");
}

/// Test graph-constrained search with variable-length paths.
///
/// Creates a hierarchical graph:
/// - org -> team_a -> project_a1, project_a2
/// - org -> team_b -> project_b1
///
/// Searches for symbols reachable from org via CONTAINS*1..3
#[test]
fn test_graph_constrained_search_variable_length() {
    let db = Database::in_memory().expect("failed to create db");

    // Create collection
    db.create_collection("projects")
        .expect("builder")
        .with_dense_vector("features", 4, DistanceMetric::Cosine)
        .build()
        .expect("create");

    let mut tx = db.begin().expect("begin");

    // Create hierarchy
    let org = tx.create_entity().expect("create").with_label("Org").with_property("name", "acme");
    let team_a =
        tx.create_entity().expect("create").with_label("Team").with_property("name", "team-a");
    let team_b =
        tx.create_entity().expect("create").with_label("Team").with_property("name", "team-b");
    let proj_a1 =
        tx.create_entity().expect("create").with_label("Project").with_property("name", "proj-a1");
    let proj_a2 =
        tx.create_entity().expect("create").with_label("Project").with_property("name", "proj-a2");
    let proj_b1 =
        tx.create_entity().expect("create").with_label("Project").with_property("name", "proj-b1");

    // External project (not in org)
    let external =
        tx.create_entity().expect("create").with_label("Project").with_property("name", "external");

    tx.put_entity(&org).expect("put");
    tx.put_entity(&team_a).expect("put");
    tx.put_entity(&team_b).expect("put");
    tx.put_entity(&proj_a1).expect("put");
    tx.put_entity(&proj_a2).expect("put");
    tx.put_entity(&proj_b1).expect("put");
    tx.put_entity(&external).expect("put");

    // Create hierarchy: org -> teams -> projects
    let e1 = tx.create_edge(org.id, team_a.id, "CONTAINS").expect("edge");
    let e2 = tx.create_edge(org.id, team_b.id, "CONTAINS").expect("edge");
    let e3 = tx.create_edge(team_a.id, proj_a1.id, "CONTAINS").expect("edge");
    let e4 = tx.create_edge(team_a.id, proj_a2.id, "CONTAINS").expect("edge");
    let e5 = tx.create_edge(team_b.id, proj_b1.id, "CONTAINS").expect("edge");

    tx.put_edge(&e1).expect("put");
    tx.put_edge(&e2).expect("put");
    tx.put_edge(&e3).expect("put");
    tx.put_edge(&e4).expect("put");
    tx.put_edge(&e5).expect("put");

    tx.commit().expect("commit");

    // Add feature vectors to projects only
    let collection = db.collection("projects").expect("collection");

    // proj_a1: ML-focused project
    collection
        .upsert_point(
            PointStruct::new(proj_a1.id.as_u64())
                .with_payload(json!({"name": "proj-a1", "focus": "ml"}))
                .with_vector("features", vec![1.0, 0.0, 0.0, 0.0]),
        )
        .expect("insert");

    // proj_a2: Data-focused project
    collection
        .upsert_point(
            PointStruct::new(proj_a2.id.as_u64())
                .with_payload(json!({"name": "proj-a2", "focus": "data"}))
                .with_vector("features", vec![0.5, 0.5, 0.0, 0.0]),
        )
        .expect("insert");

    // proj_b1: Infrastructure project
    collection
        .upsert_point(
            PointStruct::new(proj_b1.id.as_u64())
                .with_payload(json!({"name": "proj-b1", "focus": "infra"}))
                .with_vector("features", vec![0.3, 0.3, 0.3, 0.0]),
        )
        .expect("insert");

    // external: Very similar to proj_a1, but not in org
    collection
        .upsert_point(
            PointStruct::new(external.id.as_u64())
                .with_payload(json!({"name": "external", "focus": "ml"}))
                .with_vector("features", vec![0.95, 0.05, 0.0, 0.0]),
        )
        .expect("insert");

    // Search for ML-focused projects within org (requires variable-length path)
    let results = db
        .search("projects", "features")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .within_traversal(org.id, |p| p.edge_out("CONTAINS").variable_length(1, 3))
        .limit(10)
        .execute()
        .expect("search");

    // Should return org entities reachable at depths 1-3:
    // - depth 1: team_a, team_b (but not in collection)
    // - depth 2: proj_a1, proj_a2, proj_b1 (in collection)
    // External should be excluded
    let result_ids: Vec<EntityId> = results.iter().map(|r| r.entity.id).collect();

    // We expect exactly 3 projects (the ones in the collection that are reachable)
    assert_eq!(results.len(), 3, "should find exactly 3 projects in org");
    assert!(result_ids.contains(&proj_a1.id), "should contain proj_a1");
    assert!(result_ids.contains(&proj_a2.id), "should contain proj_a2");
    assert!(result_ids.contains(&proj_b1.id), "should contain proj_b1");
    assert!(!result_ids.contains(&external.id), "should NOT contain external");

    // proj_a1 should be first (closest to query)
    assert_eq!(results[0].entity.id, proj_a1.id);
}

/// Test graph-constrained search with property filters combined.
#[test]
fn test_graph_constrained_search_with_filter() {
    let db = Database::in_memory().expect("failed to create db");

    // Create collection
    db.create_collection("items")
        .expect("builder")
        .with_dense_vector("vec", 4, DistanceMetric::Cosine)
        .build()
        .expect("create");

    let mut tx = db.begin().expect("begin");

    let root = tx.create_entity().expect("create").with_label("Root");
    let item1 =
        tx.create_entity().expect("create").with_label("Item").with_property("status", "active");
    let item2 =
        tx.create_entity().expect("create").with_label("Item").with_property("status", "archived");
    let item3 =
        tx.create_entity().expect("create").with_label("Item").with_property("status", "active");
    let item4 =
        tx.create_entity().expect("create").with_label("Item").with_property("status", "active");

    tx.put_entity(&root).expect("put");
    tx.put_entity(&item1).expect("put");
    tx.put_entity(&item2).expect("put");
    tx.put_entity(&item3).expect("put");
    tx.put_entity(&item4).expect("put");

    // root contains item1, item2, item3 but NOT item4
    let e1 = tx.create_edge(root.id, item1.id, "CONTAINS").expect("edge");
    let e2 = tx.create_edge(root.id, item2.id, "CONTAINS").expect("edge");
    let e3 = tx.create_edge(root.id, item3.id, "CONTAINS").expect("edge");

    tx.put_edge(&e1).expect("put");
    tx.put_edge(&e2).expect("put");
    tx.put_edge(&e3).expect("put");

    tx.commit().expect("commit");

    // Add vectors
    let collection = db.collection("items").expect("collection");

    collection
        .upsert_point(
            PointStruct::new(item1.id.as_u64())
                .with_payload(json!({"status": "active"}))
                .with_vector("vec", vec![1.0, 0.0, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(item2.id.as_u64())
                .with_payload(json!({"status": "archived"}))
                .with_vector("vec", vec![0.9, 0.1, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(item3.id.as_u64())
                .with_payload(json!({"status": "active"}))
                .with_vector("vec", vec![0.5, 0.5, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(item4.id.as_u64())
                .with_payload(json!({"status": "active"}))
                .with_vector("vec", vec![0.95, 0.0, 0.0, 0.0]),
        )
        .expect("insert");

    // Search with both traversal constraint AND property filter
    let results = db
        .search("items", "vec")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .within_traversal(root.id, |p| p.edge_out("CONTAINS"))
        .filter(manifoldb::Filter::eq("status", "active"))
        .limit(10)
        .execute()
        .expect("search");

    // Should only return active items within root
    // item2 is archived, item4 is not in root
    let result_ids: Vec<EntityId> = results.iter().map(|r| r.entity.id).collect();

    assert_eq!(results.len(), 2, "should find 2 active items in root");
    assert!(result_ids.contains(&item1.id), "should contain item1");
    assert!(result_ids.contains(&item3.id), "should contain item3");
    assert!(!result_ids.contains(&item2.id), "should NOT contain item2 (archived)");
    assert!(!result_ids.contains(&item4.id), "should NOT contain item4 (not in root)");
}

/// Test that search without traversal constraint works normally.
#[test]
fn test_search_without_traversal_constraint() {
    let db = Database::in_memory().expect("failed to create db");

    db.create_collection("docs")
        .expect("builder")
        .with_dense_vector("emb", 4, DistanceMetric::DotProduct)
        .build()
        .expect("create");

    let collection = db.collection("docs").expect("collection");

    collection
        .upsert_point(
            PointStruct::new(1)
                .with_payload(json!({"title": "doc1"}))
                .with_vector("emb", vec![1.0, 0.0, 0.0, 0.0]),
        )
        .expect("insert");

    collection
        .upsert_point(
            PointStruct::new(2)
                .with_payload(json!({"title": "doc2"}))
                .with_vector("emb", vec![0.5, 0.5, 0.0, 0.0]),
        )
        .expect("insert");

    // Search without any traversal constraint
    let results = db
        .search("docs", "emb")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .limit(10)
        .execute()
        .expect("search");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].entity.id, EntityId::new(1));
}

/// Test graph-constrained search when no results match the traversal.
#[test]
fn test_graph_constrained_search_empty_traversal() {
    let db = Database::in_memory().expect("failed to create db");

    db.create_collection("items")
        .expect("builder")
        .with_dense_vector("vec", 4, DistanceMetric::Cosine)
        .build()
        .expect("create");

    let mut tx = db.begin().expect("begin");

    let isolated = tx.create_entity().expect("create").with_label("Root");
    let item = tx.create_entity().expect("create").with_label("Item");

    tx.put_entity(&isolated).expect("put");
    tx.put_entity(&item).expect("put");

    // No edges from isolated - it has no outgoing CONTAINS edges
    tx.commit().expect("commit");

    let collection = db.collection("items").expect("collection");

    collection
        .upsert_point(
            PointStruct::new(item.id.as_u64())
                .with_payload(json!({"name": "item"}))
                .with_vector("vec", vec![1.0, 0.0, 0.0, 0.0]),
        )
        .expect("insert");

    // Search within isolated (which has no children)
    let results = db
        .search("items", "vec")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .within_traversal(isolated.id, |p| p.edge_out("CONTAINS"))
        .limit(10)
        .execute()
        .expect("search");

    // Should return empty results since no entities are reachable
    assert!(results.is_empty(), "should return no results when traversal yields nothing");
}

/// Test graph-constrained search respects the limit parameter.
#[test]
fn test_graph_constrained_search_respects_limit() {
    let db = Database::in_memory().expect("failed to create db");

    db.create_collection("items")
        .expect("builder")
        .with_dense_vector("vec", 4, DistanceMetric::Cosine)
        .build()
        .expect("create");

    let mut tx = db.begin().expect("begin");

    let root = tx.create_entity().expect("create").with_label("Root");
    tx.put_entity(&root).expect("put");

    // Create 10 items under root
    let mut items = Vec::new();
    for _ in 0..10 {
        let item = tx.create_entity().expect("create").with_label("Item");
        tx.put_entity(&item).expect("put");
        let edge = tx.create_edge(root.id, item.id, "CONTAINS").expect("edge");
        tx.put_edge(&edge).expect("put");
        items.push(item);
    }

    tx.commit().expect("commit");

    let collection = db.collection("items").expect("collection");

    for (i, item) in items.iter().enumerate() {
        collection
            .upsert_point(
                PointStruct::new(item.id.as_u64())
                    .with_payload(json!({"index": i}))
                    .with_vector("vec", vec![(10 - i) as f32 / 10.0, 0.0, 0.0, 0.0]),
            )
            .expect("insert");
    }

    // Search with limit of 3
    let results = db
        .search("items", "vec")
        .expect("search builder")
        .query(vec![1.0, 0.0, 0.0, 0.0])
        .within_traversal(root.id, |p| p.edge_out("CONTAINS"))
        .limit(3)
        .execute()
        .expect("search");

    assert_eq!(results.len(), 3, "should respect limit of 3");
}
