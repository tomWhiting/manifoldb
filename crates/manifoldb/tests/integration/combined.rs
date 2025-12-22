//! Combined multi-paradigm query integration tests.
//!
//! Tests that exercise graph, vector, and relational features together,
//! simulating real-world use cases.

use std::collections::HashSet;

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Use Case: Document Management System
// ============================================================================

/// Documents with:
/// - Graph relationships (folder hierarchy, related documents)
/// - Vector embeddings for semantic search
/// - Properties for metadata
#[test]
fn test_document_management_system() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create folders
    let root = tx
        .create_entity()
        .expect("failed")
        .with_label("Folder")
        .with_property("name", "Root")
        .with_property("path", "/");
    let docs = tx
        .create_entity()
        .expect("failed")
        .with_label("Folder")
        .with_property("name", "Documents")
        .with_property("path", "/Documents");
    let projects = tx
        .create_entity()
        .expect("failed")
        .with_label("Folder")
        .with_property("name", "Projects")
        .with_property("path", "/Projects");

    tx.put_entity(&root).expect("failed");
    tx.put_entity(&docs).expect("failed");
    tx.put_entity(&projects).expect("failed");

    // Folder hierarchy
    let e1 = tx.create_edge(root.id, docs.id, "CONTAINS").expect("failed");
    let e2 = tx.create_edge(root.id, projects.id, "CONTAINS").expect("failed");
    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");

    // Create documents with embeddings (simulated)
    let doc1 = tx
        .create_entity()
        .expect("failed")
        .with_label("Document")
        .with_property("title", "Project Proposal")
        .with_property("author", "Alice")
        .with_property("embedding", vec![0.1f32, 0.2, 0.3, 0.4]); // Simplified embedding

    let doc2 = tx
        .create_entity()
        .expect("failed")
        .with_label("Document")
        .with_property("title", "Technical Spec")
        .with_property("author", "Bob")
        .with_property("embedding", vec![0.2f32, 0.3, 0.4, 0.5]);

    let doc3 = tx
        .create_entity()
        .expect("failed")
        .with_label("Document")
        .with_property("title", "Meeting Notes")
        .with_property("author", "Alice")
        .with_property("embedding", vec![0.15f32, 0.25, 0.35, 0.45]);

    tx.put_entity(&doc1).expect("failed");
    tx.put_entity(&doc2).expect("failed");
    tx.put_entity(&doc3).expect("failed");

    // Documents in folders
    let e3 = tx.create_edge(projects.id, doc1.id, "CONTAINS").expect("failed");
    let e4 = tx.create_edge(projects.id, doc2.id, "CONTAINS").expect("failed");
    let e5 = tx.create_edge(docs.id, doc3.id, "CONTAINS").expect("failed");
    tx.put_edge(&e3).expect("failed");
    tx.put_edge(&e4).expect("failed");
    tx.put_edge(&e5).expect("failed");

    // Related documents
    let e6 = tx.create_edge(doc1.id, doc2.id, "RELATED_TO").expect("failed");
    tx.put_edge(&e6).expect("failed");

    tx.commit().expect("failed to commit");

    // Query: Find all documents in Projects folder
    let tx = db.begin_read().expect("failed to begin read");
    let project_contents = tx.get_outgoing_edges(projects.id).expect("failed");

    let doc_ids: Vec<_> = project_contents
        .iter()
        .filter(|e| e.edge_type.as_str() == "CONTAINS")
        .map(|e| e.target)
        .collect();

    assert_eq!(doc_ids.len(), 2);

    // Query: Find documents by author
    let mut alice_docs = 0;
    for id in &doc_ids {
        let doc = tx.get_entity(*id).expect("failed").expect("not found");
        if doc.get_property("author") == Some(&Value::String("Alice".to_string())) {
            alice_docs += 1;
        }
    }
    assert_eq!(alice_docs, 1);

    // Query: Find related documents
    let related = tx.get_outgoing_edges(doc1.id).expect("failed");
    let related_docs: Vec<_> =
        related.iter().filter(|e| e.edge_type.as_str() == "RELATED_TO").collect();
    assert_eq!(related_docs.len(), 1);
    assert_eq!(related_docs[0].target, doc2.id);
}

// ============================================================================
// Use Case: Social Network with Recommendations
// ============================================================================

/// Users with:
/// - Social graph (friends, follows)
/// - Interest embeddings for recommendations
/// - Profile properties
#[test]
fn test_social_network_recommendations() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create users with interests (embeddings would represent interest vectors)
    let alice = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("username", "alice")
        .with_property("interests", vec![0.8f32, 0.2, 0.5, 0.1]); // High tech, low sports

    let bob = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("username", "bob")
        .with_property("interests", vec![0.7f32, 0.3, 0.4, 0.2]); // Similar to Alice

    let charlie = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("username", "charlie")
        .with_property("interests", vec![0.2f32, 0.9, 0.1, 0.8]); // Different interests

    let diana = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("username", "diana")
        .with_property("interests", vec![0.75f32, 0.25, 0.45, 0.15]); // Similar to Alice

    tx.put_entity(&alice).expect("failed");
    tx.put_entity(&bob).expect("failed");
    tx.put_entity(&charlie).expect("failed");
    tx.put_entity(&diana).expect("failed");

    // Social connections
    let f1 = tx.create_edge(alice.id, bob.id, "FRIENDS").expect("failed");
    let f2 = tx.create_edge(alice.id, charlie.id, "FOLLOWS").expect("failed");
    tx.put_edge(&f1).expect("failed");
    tx.put_edge(&f2).expect("failed");

    tx.commit().expect("failed");

    // Query: Find Alice's friends
    let tx = db.begin_read().expect("failed");
    let alice_connections = tx.get_outgoing_edges(alice.id).expect("failed");
    let friends: Vec<_> =
        alice_connections.iter().filter(|e| e.edge_type.as_str() == "FRIENDS").collect();
    assert_eq!(friends.len(), 1);
    assert_eq!(friends[0].target, bob.id);

    // Query: Find friends of friends (2-hop)
    let mut friends_of_friends = HashSet::new();
    for friend_edge in &friends {
        let fof_edges = tx.get_outgoing_edges(friend_edge.target).expect("failed");
        for edge in fof_edges {
            if edge.edge_type.as_str() == "FRIENDS" && edge.target != alice.id {
                friends_of_friends.insert(edge.target);
            }
        }
    }

    // Extract interest vectors for similarity matching
    let alice_entity = tx.get_entity(alice.id).expect("failed").expect("not found");
    let alice_interests = match alice_entity.get_property("interests") {
        Some(Value::Vector(v)) => v.clone(),
        _ => panic!("missing interests"),
    };

    // Get all users and their interests for similarity
    let users = [bob.id, charlie.id, diana.id];
    let mut user_similarities = Vec::new();

    for user_id in users {
        let user = tx.get_entity(user_id).expect("failed").expect("not found");
        if let Some(Value::Vector(interests)) = user.get_property("interests") {
            // Calculate cosine similarity (simplified)
            let dot: f32 = alice_interests.iter().zip(interests.iter()).map(|(a, b)| a * b).sum();
            user_similarities.push((user_id, dot));
        }
    }

    // Sort by similarity
    user_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Bob and Diana should be most similar to Alice
    assert!(
        user_similarities[0].0 == bob.id || user_similarities[0].0 == diana.id,
        "Bob or Diana should be most similar"
    );
}

// ============================================================================
// Use Case: E-commerce Product Search
// ============================================================================

/// Products with:
/// - Category hierarchy (graph)
/// - Product embeddings for similar items
/// - Price, rating, stock properties
#[test]
fn test_ecommerce_product_search() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create categories
    let electronics = tx
        .create_entity()
        .expect("failed")
        .with_label("Category")
        .with_property("name", "Electronics");
    let phones =
        tx.create_entity().expect("failed").with_label("Category").with_property("name", "Phones");
    let laptops =
        tx.create_entity().expect("failed").with_label("Category").with_property("name", "Laptops");

    tx.put_entity(&electronics).expect("failed");
    tx.put_entity(&phones).expect("failed");
    tx.put_entity(&laptops).expect("failed");

    // Category hierarchy
    let c1 = tx.create_edge(electronics.id, phones.id, "HAS_SUBCATEGORY").expect("failed");
    let c2 = tx.create_edge(electronics.id, laptops.id, "HAS_SUBCATEGORY").expect("failed");
    tx.put_edge(&c1).expect("failed");
    tx.put_edge(&c2).expect("failed");

    // Create products
    let products = vec![
        ("iPhone 15", 999.0, 4.5, 100, phones.id, vec![0.9f32, 0.8, 0.1, 0.2]),
        ("Galaxy S24", 899.0, 4.4, 150, phones.id, vec![0.88f32, 0.78, 0.12, 0.22]),
        ("MacBook Pro", 1999.0, 4.8, 50, laptops.id, vec![0.1f32, 0.2, 0.9, 0.8]),
        ("ThinkPad X1", 1799.0, 4.6, 75, laptops.id, vec![0.12f32, 0.22, 0.88, 0.78]),
    ];

    let mut product_ids = Vec::new();
    for (name, price, rating, stock, category_id, embedding) in products {
        let product = tx
            .create_entity()
            .expect("failed")
            .with_label("Product")
            .with_property("name", name)
            .with_property("price", price)
            .with_property("rating", rating)
            .with_property("stock", stock as i64)
            .with_property("embedding", embedding);

        product_ids.push(product.id);
        tx.put_entity(&product).expect("failed");

        let edge = tx.create_edge(category_id, product.id, "CONTAINS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Query: Find all phones under $950
    let tx = db.begin_read().expect("failed");
    let phone_edges = tx.get_outgoing_edges(phones.id).expect("failed");

    let mut affordable_phones = Vec::new();
    for edge in phone_edges {
        if edge.edge_type.as_str() == "CONTAINS" {
            let product = tx.get_entity(edge.target).expect("failed").expect("not found");
            if let Some(Value::Float(price)) = product.get_property("price") {
                if *price < 950.0 {
                    affordable_phones.push(product);
                }
            }
        }
    }

    assert_eq!(affordable_phones.len(), 1);
    assert_eq!(
        affordable_phones[0].get_property("name"),
        Some(&Value::String("Galaxy S24".to_string()))
    );

    // Query: Find products with rating >= 4.5
    let mut high_rated = Vec::new();
    for &id in &product_ids {
        let product = tx.get_entity(id).expect("failed").expect("not found");
        if let Some(Value::Float(rating)) = product.get_property("rating") {
            if *rating >= 4.5 {
                high_rated.push(product);
            }
        }
    }

    assert_eq!(high_rated.len(), 3); // iPhone, MacBook, ThinkPad
}

// ============================================================================
// Use Case: Knowledge Graph with Semantic Search
// ============================================================================

/// Entities representing concepts with:
/// - Relationships (is_a, part_of, related_to)
/// - Concept embeddings
/// - Descriptions and metadata
#[test]
fn test_knowledge_graph_semantic() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    // Create concepts
    let animal = tx
        .create_entity()
        .expect("failed")
        .with_label("Concept")
        .with_property("name", "Animal")
        .with_property("embedding", vec![0.5f32, 0.5, 0.0, 0.0]);

    let mammal = tx
        .create_entity()
        .expect("failed")
        .with_label("Concept")
        .with_property("name", "Mammal")
        .with_property("embedding", vec![0.6f32, 0.4, 0.1, 0.0]);

    let dog = tx
        .create_entity()
        .expect("failed")
        .with_label("Concept")
        .with_property("name", "Dog")
        .with_property("embedding", vec![0.7f32, 0.3, 0.2, 0.1]);

    let cat = tx
        .create_entity()
        .expect("failed")
        .with_label("Concept")
        .with_property("name", "Cat")
        .with_property("embedding", vec![0.65f32, 0.35, 0.15, 0.05]);

    let pet = tx
        .create_entity()
        .expect("failed")
        .with_label("Concept")
        .with_property("name", "Pet")
        .with_property("embedding", vec![0.4f32, 0.2, 0.3, 0.1]);

    tx.put_entity(&animal).expect("failed");
    tx.put_entity(&mammal).expect("failed");
    tx.put_entity(&dog).expect("failed");
    tx.put_entity(&cat).expect("failed");
    tx.put_entity(&pet).expect("failed");

    // Create relationships
    let r1 = tx.create_edge(mammal.id, animal.id, "IS_A").expect("failed");
    let r2 = tx.create_edge(dog.id, mammal.id, "IS_A").expect("failed");
    let r3 = tx.create_edge(cat.id, mammal.id, "IS_A").expect("failed");
    let r4 = tx.create_edge(dog.id, pet.id, "CAN_BE").expect("failed");
    let r5 = tx.create_edge(cat.id, pet.id, "CAN_BE").expect("failed");

    tx.put_edge(&r1).expect("failed");
    tx.put_edge(&r2).expect("failed");
    tx.put_edge(&r3).expect("failed");
    tx.put_edge(&r4).expect("failed");
    tx.put_edge(&r5).expect("failed");

    tx.commit().expect("failed");

    // Query: Traverse IS_A hierarchy from Dog to find ancestors
    let tx = db.begin_read().expect("failed");
    let mut ancestors = Vec::new();
    let mut current = dog.id;

    loop {
        let edges = tx.get_outgoing_edges(current).expect("failed");
        let is_a: Vec<_> = edges.iter().filter(|e| e.edge_type.as_str() == "IS_A").collect();

        if is_a.is_empty() {
            break;
        }

        current = is_a[0].target;
        let ancestor = tx.get_entity(current).expect("failed").expect("not found");
        if let Some(Value::String(name)) = ancestor.get_property("name") {
            ancestors.push(name.clone());
        }
    }

    assert_eq!(ancestors, vec!["Mammal", "Animal"]);

    // Query: Find all concepts that CAN_BE a Pet
    let pet_edges = tx.get_incoming_edges(pet.id).expect("failed");
    let can_be_pets: Vec<_> =
        pet_edges.iter().filter(|e| e.edge_type.as_str() == "CAN_BE").map(|e| e.source).collect();

    assert_eq!(can_be_pets.len(), 2);
    assert!(can_be_pets.contains(&dog.id));
    assert!(can_be_pets.contains(&cat.id));
}

// ============================================================================
// Use Case: Mixed Workload Simulation
// ============================================================================

/// Simulates a mixed read/write workload with interleaved operations
#[test]
fn test_mixed_workload() {
    let db = Database::in_memory().expect("failed to create db");

    // Phase 1: Create base data
    let mut entity_ids = Vec::new();
    {
        let mut tx = db.begin().expect("failed");
        for i in 0..50 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("Item")
                .with_property("index", i as i64)
                .with_property("value", (i * 100) as i64);
            entity_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Phase 2: Read and verify
    {
        let tx = db.begin_read().expect("failed");
        for (i, &id) in entity_ids.iter().enumerate() {
            let entity = tx.get_entity(id).expect("failed").expect("not found");
            assert_eq!(entity.get_property("index"), Some(&Value::Int(i as i64)));
        }
    }

    // Phase 3: Update half the entities
    {
        let mut tx = db.begin().expect("failed");
        for (i, &id) in entity_ids.iter().enumerate() {
            if i % 2 == 0 {
                let mut entity = tx.get_entity(id).expect("failed").expect("not found");
                entity.set_property("updated", true);
                entity.set_property("value", (i * 200) as i64);
                tx.put_entity(&entity).expect("failed");
            }
        }
        tx.commit().expect("failed");
    }

    // Phase 4: Add edges between consecutive entities
    {
        let mut tx = db.begin().expect("failed");
        for i in 0..(entity_ids.len() - 1) {
            let edge = tx.create_edge(entity_ids[i], entity_ids[i + 1], "NEXT").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Phase 5: Verify final state
    {
        let tx = db.begin_read().expect("failed");

        // Check updates
        for (i, &id) in entity_ids.iter().enumerate() {
            let entity = tx.get_entity(id).expect("failed").expect("not found");
            if i % 2 == 0 {
                assert_eq!(entity.get_property("updated"), Some(&Value::Bool(true)));
                assert_eq!(entity.get_property("value"), Some(&Value::Int((i * 200) as i64)));
            } else {
                assert!(entity.get_property("updated").is_none());
            }
        }

        // Check edges
        for i in 0..(entity_ids.len() - 1) {
            let edges = tx.get_outgoing_edges(entity_ids[i]).expect("failed");
            assert!(!edges.is_empty(), "entity {i} should have outgoing edges");
            assert_eq!(edges[0].target, entity_ids[i + 1]);
        }
    }
}

// ============================================================================
// Use Case: Vector Search with Graph Filtering
// ============================================================================

/// Search for similar items, then filter by graph relationships
#[test]
fn test_vector_search_with_graph_filter() {
    // Create graph database
    let db = Database::in_memory().expect("failed to create db");

    // Create category structure
    let mut tx = db.begin().expect("failed");

    let tech =
        tx.create_entity().expect("failed").with_label("Category").with_property("name", "Tech");
    let books =
        tx.create_entity().expect("failed").with_label("Category").with_property("name", "Books");

    tx.put_entity(&tech).expect("failed");
    tx.put_entity(&books).expect("failed");

    // Create items with embeddings stored in properties
    let items = vec![
        ("Laptop", tech.id, vec![0.9f32, 0.1, 0.8, 0.2]),
        ("Phone", tech.id, vec![0.85f32, 0.15, 0.75, 0.25]),
        ("Tablet", tech.id, vec![0.88f32, 0.12, 0.78, 0.22]),
        ("Novel", books.id, vec![0.1f32, 0.9, 0.2, 0.8]),
        ("Textbook", books.id, vec![0.2f32, 0.8, 0.3, 0.7]),
    ];

    let mut item_ids = Vec::new();
    for (name, category_id, embedding) in items {
        let item = tx
            .create_entity()
            .expect("failed")
            .with_label("Item")
            .with_property("name", name)
            .with_property("embedding", embedding);
        item_ids.push(item.id);
        tx.put_entity(&item).expect("failed");

        let edge = tx.create_edge(category_id, item.id, "CONTAINS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Query: Find items similar to "Laptop" but only in Tech category
    let tx = db.begin_read().expect("failed");

    // Get tech category items
    let tech_edges = tx.get_outgoing_edges(tech.id).expect("failed");
    let tech_item_ids: HashSet<_> = tech_edges
        .iter()
        .filter(|e| e.edge_type.as_str() == "CONTAINS")
        .map(|e| e.target)
        .collect();

    // Get all items with embeddings
    let mut tech_items_with_embeddings = Vec::new();
    for &id in &tech_item_ids {
        let item = tx.get_entity(id).expect("failed").expect("not found");
        if let Some(Value::Vector(emb)) = item.get_property("embedding") {
            tech_items_with_embeddings.push((id, emb.clone()));
        }
    }

    // Query embedding (similar to laptop)
    let query_embedding = vec![0.9f32, 0.1, 0.8, 0.2];

    // Find most similar within tech items only
    let mut best_match = None;
    let mut best_distance = f32::MAX;

    for (id, emb) in &tech_items_with_embeddings {
        let distance: f32 = query_embedding
            .iter()
            .zip(emb.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        if distance < best_distance {
            best_distance = distance;
            best_match = Some(*id);
        }
    }

    // The laptop itself should be the best match
    assert_eq!(best_match, Some(item_ids[0])); // Laptop
}

// ============================================================================
// Concurrent Operation Patterns
// ============================================================================

#[test]
fn test_multiple_read_transactions() {
    let db = Database::in_memory().expect("failed to create db");

    // Create data
    {
        let mut tx = db.begin().expect("failed");
        for i in 0..10 {
            let entity = tx.create_entity().expect("failed").with_property("id", i as i64);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Multiple concurrent reads should work
    let tx1 = db.begin_read().expect("failed");
    let tx2 = db.begin_read().expect("failed");
    let tx3 = db.begin_read().expect("failed");

    // All should see the same data
    let e1 = tx1.get_entity(EntityId::new(1)).expect("failed");
    let e2 = tx2.get_entity(EntityId::new(1)).expect("failed");
    let e3 = tx3.get_entity(EntityId::new(1)).expect("failed");

    assert!(e1.is_some());
    assert!(e2.is_some());
    assert!(e3.is_some());

    // Clean up
    drop(tx1);
    drop(tx2);
    drop(tx3);
}
