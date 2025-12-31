//! ManifoldDB Test Drive
//!
//! A comprehensive interactive test of all major features.
//! Run with: cargo run --example test_drive

#![allow(unused_variables)]
#![allow(clippy::unreadable_literal)]

use manifoldb::collection::{DistanceMetric, Filter, PointStruct};
use manifoldb::{Database, EntityId};
use serde_json::json;

const SEP_MAJOR: &str = "============================================================";
const SEP_MINOR: &str = "------------------------------------------------------------";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ManifoldDB Test Drive\n");
    println!("{}", SEP_MAJOR);

    // Create in-memory database
    let db = Database::in_memory()?;
    println!("✓ Created in-memory database\n");

    // =========================================================================
    // PART 1: Collection API (Qdrant-style)
    // =========================================================================
    println!("PART 1: Collection API (Qdrant-style)");
    println!("{}", SEP_MINOR);

    // Create a documents collection with dense vectors
    let docs = db
        .create_collection("documents")?
        .with_dense_vector("embedding", 128, DistanceMetric::Cosine)
        .build()?;
    println!("✓ Created 'documents' collection with 128-dim dense vector");

    // Create a products collection with multiple vector types
    let products = db
        .create_collection("products")?
        .with_dense_vector("text_embedding", 64, DistanceMetric::Cosine)
        .with_dense_vector("image_embedding", 128, DistanceMetric::DotProduct)
        .build()?;
    println!("✓ Created 'products' collection with text + image embeddings");

    // Insert documents with payloads
    let doc_titles = [
        "Introduction to Rust Programming",
        "Advanced Python Techniques",
        "Machine Learning Fundamentals",
        "Database Design Patterns",
        "Rust Async Programming Guide",
    ];

    for (i, title) in doc_titles.iter().enumerate() {
        // Generate a pseudo-embedding based on content
        let mut embedding = vec![0.0f32; 128];
        if title.contains("Rust") {
            embedding[0] = 0.9;
            embedding[1] = 0.8;
        }
        if title.contains("Python") {
            embedding[2] = 0.9;
            embedding[3] = 0.7;
        }
        if title.contains("Machine Learning") {
            embedding[4] = 0.95;
            embedding[5] = 0.85;
        }
        if title.contains("Database") {
            embedding[6] = 0.88;
            embedding[7] = 0.75;
        }
        embedding[10] = (i as f32) / 10.0; // Unique component

        docs.upsert_point(
            PointStruct::new((i + 1) as u64)
                .with_payload(json!({
                    "title": title,
                    "category": if title.contains("Rust") { "rust" } else { "other" },
                    "pages": 100 + i * 50
                }))
                .with_vector("embedding", embedding),
        )?;
    }
    println!("✓ Inserted {} documents with embeddings", doc_titles.len());

    // Search for Rust-related documents
    let rust_query = {
        let mut v = vec![0.0f32; 128];
        v[0] = 0.9;
        v[1] = 0.8;
        v
    };

    let results =
        docs.search("embedding").query(rust_query).limit(3).with_payload(true).execute()?;

    println!("\n📚 Search results for 'Rust-like' embedding:");
    for (i, r) in results.iter().enumerate() {
        let title = r.payload.as_ref().and_then(|p| p["title"].as_str()).unwrap_or("Unknown");
        println!("   {}. {} (score: {:.4})", i + 1, title, r.score);
    }

    // Search with filter
    let filtered_results = docs
        .search("embedding")
        .query(vec![0.5; 128])
        .limit(10)
        .filter(Filter::eq("category", "rust"))
        .with_payload(true)
        .execute()?;

    println!("\n📚 Filtered search (category = 'rust'):");
    for r in &filtered_results {
        let title = r.payload.as_ref().and_then(|p| p["title"].as_str()).unwrap_or("Unknown");
        println!("   - {}", title);
    }

    // Test point operations
    let payload = docs.get_payload(1.into())?;
    println!("\n✓ Retrieved payload for point 1: {:?}", payload.map(|p| p["title"].clone()));

    docs.update_payload(
        1.into(),
        json!({"title": "Intro to Rust (Updated)", "category": "rust", "pages": 150}),
    )?;
    println!("✓ Updated payload for point 1");

    let updated = docs.get_payload(1.into())?;
    println!("✓ Verified update: {:?}", updated.map(|p| p["title"].clone()));

    println!("\n✓ Collection API test complete!\n");

    // =========================================================================
    // PART 2: SQL Interface
    // =========================================================================
    println!("PART 2: SQL Interface");
    println!("{}", SEP_MINOR);

    // Create table
    db.execute("CREATE TABLE users (id BIGINT, name TEXT, email TEXT, age INT)")?;
    println!("✓ Created users table");

    // Insert data
    db.execute(
        "INSERT INTO users (id, name, email, age) VALUES (1, 'Alice', 'alice@example.com', 30)",
    )?;
    db.execute(
        "INSERT INTO users (id, name, email, age) VALUES (2, 'Bob', 'bob@example.com', 25)",
    )?;
    db.execute(
        "INSERT INTO users (id, name, email, age) VALUES (3, 'Charlie', 'charlie@example.com', 35)",
    )?;
    db.execute(
        "INSERT INTO users (id, name, email, age) VALUES (4, 'Diana', 'diana@example.com', 28)",
    )?;
    db.execute(
        "INSERT INTO users (id, name, email, age) VALUES (5, 'Eve', 'eve@example.com', 32)",
    )?;
    println!("✓ Inserted 5 users");

    // Query data
    let result = db.query("SELECT name, age FROM users WHERE age > 28 ORDER BY age DESC")?;
    println!("\n👥 Users over 28 (ordered by age desc):");
    for row in result.rows() {
        println!("   - {:?}", row);
    }

    // Aggregations
    let result = db.query("SELECT COUNT(*) as count, AVG(age) as avg_age FROM users")?;
    println!("\n📊 Aggregations:");
    for row in result.rows() {
        println!("   Count: {:?}, Avg Age: {:?}", row.get(0), row.get(1));
    }

    // Update
    db.execute("UPDATE users SET age = 31 WHERE name = 'Alice'")?;
    let result = db.query("SELECT name, age FROM users WHERE name = 'Alice'")?;
    println!("\n✓ Updated Alice's age: {:?}", result.rows().first());

    // LIKE query
    let result = db.query("SELECT name FROM users WHERE email LIKE '%example.com'")?;
    println!("\n✓ LIKE query found {} users with example.com emails", result.len());

    println!("\n✓ SQL Interface test complete!\n");

    // =========================================================================
    // PART 3: Graph Operations
    // =========================================================================
    println!("PART 3: Graph Operations");
    println!("{}", SEP_MINOR);

    let mut tx = db.begin()?;

    // Create people
    let alice = tx
        .create_entity()?
        .with_label("Person")
        .with_property("name", "Alice")
        .with_property("role", "Engineer");
    tx.put_entity(&alice)?;

    let bob = tx
        .create_entity()?
        .with_label("Person")
        .with_property("name", "Bob")
        .with_property("role", "Manager");
    tx.put_entity(&bob)?;

    let charlie = tx
        .create_entity()?
        .with_label("Person")
        .with_property("name", "Charlie")
        .with_property("role", "Designer");
    tx.put_entity(&charlie)?;

    // Create project
    let project = tx
        .create_entity()?
        .with_label("Project")
        .with_property("name", "ManifoldDB")
        .with_property("status", "active");
    tx.put_entity(&project)?;

    println!("✓ Created 3 Person entities and 1 Project entity");

    // Create relationships
    let e1 = tx.create_edge(alice.id, bob.id, "REPORTS_TO")?;
    tx.put_edge(&e1)?;

    let e2 = tx.create_edge(charlie.id, bob.id, "REPORTS_TO")?;
    tx.put_edge(&e2)?;

    let e3 =
        tx.create_edge(alice.id, project.id, "WORKS_ON")?.with_property("hours_per_week", 40i64);
    tx.put_edge(&e3)?;

    let e4 = tx.create_edge(bob.id, project.id, "MANAGES")?;
    tx.put_edge(&e4)?;

    let e5 =
        tx.create_edge(charlie.id, project.id, "WORKS_ON")?.with_property("hours_per_week", 20i64);
    tx.put_edge(&e5)?;

    tx.commit()?;
    println!("✓ Created relationships: REPORTS_TO, WORKS_ON, MANAGES");

    // Query graph
    let tx = db.begin_read()?;

    // Who reports to Bob?
    let reports = tx.get_incoming_edges(bob.id)?;
    let reporters: Vec<_> =
        reports.iter().filter(|e| e.edge_type.as_str() == "REPORTS_TO").collect();
    println!("\n👥 People reporting to Bob: {} people", reporters.len());

    for edge in &reporters {
        let person = tx.get_entity(edge.source)?.unwrap();
        println!("   - {:?}", person.get_property("name"));
    }

    // What is Alice connected to?
    let alice_edges = tx.get_outgoing_edges(alice.id)?;
    println!("\n🔗 Alice's connections:");
    for edge in &alice_edges {
        let target = tx.get_entity(edge.target)?.unwrap();
        println!("   -[{}]-> {:?}", edge.edge_type.as_str(), target.get_property("name"));
    }

    drop(tx);
    println!("\n✓ Graph Operations test complete!\n");

    // =========================================================================
    // PART 4: Vector Search with HNSW Index
    // =========================================================================
    println!("PART 4: Vector Search with HNSW Index");
    println!("{}", SEP_MINOR);

    // Create table with vector column
    // Note: 'id' is a reserved column that returns the internal EntityId
    // Use 'label' as our unique identifier for updates
    db.execute("CREATE TABLE embeddings (label TEXT, vec VECTOR(4))")?;
    println!("✓ Created embeddings table with VECTOR(4) column");

    // Create HNSW index
    db.execute("CREATE INDEX emb_idx ON embeddings USING hnsw (vec)")?;
    println!("✓ Created HNSW index on vec column");

    // Insert vectors in different "clusters"
    // Cluster 1: [1, 0, 0, 0] direction
    db.execute("INSERT INTO embeddings (label, vec) VALUES ('cluster1_a', [1.0, 0.0, 0.0, 0.0])")?;
    db.execute("INSERT INTO embeddings (label, vec) VALUES ('cluster1_b', [0.9, 0.1, 0.0, 0.0])")?;
    db.execute(
        "INSERT INTO embeddings (label, vec) VALUES ('cluster1_c', [0.95, 0.05, 0.0, 0.0])",
    )?;

    // Cluster 2: [0, 1, 0, 0] direction
    db.execute("INSERT INTO embeddings (label, vec) VALUES ('cluster2_a', [0.0, 1.0, 0.0, 0.0])")?;
    db.execute("INSERT INTO embeddings (label, vec) VALUES ('cluster2_b', [0.1, 0.9, 0.0, 0.0])")?;

    // Cluster 3: [0, 0, 1, 0] direction
    db.execute("INSERT INTO embeddings (label, vec) VALUES ('cluster3_a', [0.0, 0.0, 1.0, 0.0])")?;
    db.execute("INSERT INTO embeddings (label, vec) VALUES ('cluster3_b', [0.0, 0.0, 0.9, 0.1])")?;

    println!("✓ Inserted 7 vectors in 3 clusters");

    // Vector similarity search
    let result = db
        .query("SELECT label, vec FROM embeddings ORDER BY vec <-> [1.0, 0.0, 0.0, 0.0] LIMIT 3")?;
    println!("\n🔍 Nearest to [1,0,0,0]:");
    for row in result.rows() {
        println!("   - {:?}", row);
    }

    let result = db
        .query("SELECT label, vec FROM embeddings ORDER BY vec <-> [0.0, 1.0, 0.0, 0.0] LIMIT 3")?;
    println!("\n🔍 Nearest to [0,1,0,0]:");
    for row in result.rows() {
        println!("   - {:?}", row);
    }

    // Update a vector using label and verify search results change
    db.execute("UPDATE embeddings SET vec = [0.0, 0.0, 0.0, 1.0] WHERE label = 'cluster1_a'")?;
    println!("\n✓ Updated vector for 'cluster1_a' to [0,0,0,1]");

    let result = db
        .query("SELECT label, vec FROM embeddings ORDER BY vec <-> [0.0, 0.0, 0.0, 1.0] LIMIT 3")?;
    println!("🔍 Nearest to [0,0,0,1] (should now include cluster1_a):");
    for row in result.rows() {
        println!("   - {:?}", row);
    }

    println!("\n✓ HNSW Vector Search test complete!\n");

    // =========================================================================
    // PART 5: Bulk Operations
    // =========================================================================
    println!("PART 5: Bulk Operations");
    println!("{}", SEP_MINOR);

    let mut tx = db.begin()?;

    // Bulk insert entities
    let start = std::time::Instant::now();
    let mut entities = Vec::new();
    for i in 0..1000 {
        let entity = tx
            .create_entity()?
            .with_label("BulkItem")
            .with_property("index", i as i64)
            .with_property("batch", "test_batch");
        entities.push(entity.id);
        tx.put_entity(&entity)?;
    }
    tx.commit()?;
    // Clear cache after Transaction API modifications (required for SQL queries to see changes)
    db.invalidate_cache_for_tables(&["BulkItem".to_string()]);
    println!("✓ Bulk inserted 1000 entities in {:?}", start.elapsed());

    // Verify count - use label as table name
    let result = db.query("SELECT COUNT(*) FROM BulkItem")?;
    println!("✓ Verified count: {:?}", result.rows().first());

    // Bulk update
    let start = std::time::Instant::now();
    let mut tx = db.begin()?;
    for (i, &id) in entities.iter().take(500).enumerate() {
        let mut entity = tx.get_entity(id)?.unwrap();
        entity.set_property("updated", true);
        entity.set_property("update_index", i as i64);
        tx.put_entity(&entity)?;
    }
    tx.commit()?;
    println!("✓ Bulk updated 500 entities in {:?}", start.elapsed());

    // Bulk delete
    let start = std::time::Instant::now();
    let mut tx = db.begin()?;
    for &id in entities.iter().skip(800) {
        tx.delete_entity(id)?;
    }
    tx.commit()?;
    // Clear cache after Transaction API modifications
    db.invalidate_cache_for_tables(&["BulkItem".to_string()]);
    println!("✓ Bulk deleted 200 entities in {:?}", start.elapsed());

    let result = db.query("SELECT COUNT(*) FROM BulkItem")?;
    println!("✓ Remaining count: {:?}", result.rows().first());

    println!("\n✓ Bulk Operations test complete!\n");

    // =========================================================================
    // PART 6: Edge Cases & Error Handling
    // =========================================================================
    println!("PART 6: Edge Cases & Error Handling");
    println!("{}", SEP_MINOR);

    // Try to get non-existent entity
    let tx = db.begin_read()?;
    let missing = tx.get_entity(EntityId::new(999999))?;
    assert!(missing.is_none());
    println!("✓ Non-existent entity returns None");
    drop(tx);

    // Try to open non-existent collection
    let result = db.collection("nonexistent");
    assert!(result.is_err());
    println!("✓ Non-existent collection returns error");

    // Try to search non-existent vector in collection
    let missing_vec = docs.get_vector(999.into(), "embedding")?;
    assert!(missing_vec.is_none());
    println!("✓ Non-existent point vector returns None");

    // Invalid SQL
    let result = db.execute("INVALID SQL STATEMENT");
    assert!(result.is_err());
    println!("✓ Invalid SQL returns error");

    // Dimension mismatch (try to insert wrong dimension vector via Collection API)
    let wrong_dim_result = docs.upsert_point(
        PointStruct::new(999u64).with_vector("embedding", vec![1.0; 64]), // Should be 128
    );
    // This might or might not error depending on validation - let's see
    println!("✓ Dimension mismatch handling: {:?}", wrong_dim_result.is_err());

    println!("\n✓ Edge Cases test complete!\n");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("{}", SEP_MAJOR);
    println!("🎉 TEST DRIVE COMPLETE!");
    println!("{}", SEP_MAJOR);
    println!("\nAll major features tested:");
    println!("  ✓ Collection API (create, upsert, search, filter)");
    println!("  ✓ SQL Interface (DDL, DML, queries, aggregations)");
    println!("  ✓ Graph Operations (entities, edges, traversals)");
    println!("  ✓ HNSW Vector Search (index, similarity, updates)");
    println!("  ✓ Bulk Operations (insert, update, delete)");
    println!("  ✓ Error Handling (missing data, invalid input)");
    println!("\nManifoldDB is ready for action! 🚀\n");

    Ok(())
}
