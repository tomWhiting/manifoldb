//! End-to-end integration tests for ManifoldDB.
//!
//! These tests exercise complete workflows combining graph, vector, and SQL
//! operations, simulating real embedded database scenarios.

use std::collections::{HashMap, HashSet};

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Embedded IoT Device Database
// ============================================================================

/// Simulates an IoT device storing sensor data locally:
/// - Graph: Device topology and sensor relationships
/// - Vector: Anomaly detection embeddings
/// - Properties: Sensor readings and timestamps
#[test]
fn test_iot_sensor_database() {
    let db = Database::in_memory().expect("failed to create db");

    // Setup phase: Create device topology
    let mut tx = db.begin().expect("failed to begin");

    // Create gateway device
    let gateway = tx
        .create_entity()
        .expect("failed")
        .with_label("Device")
        .with_property("type", "gateway")
        .with_property("mac", "AA:BB:CC:DD:EE:FF")
        .with_property("location", "main_building");
    tx.put_entity(&gateway).expect("failed");

    // Create sensors connected to gateway
    let mut sensor_ids = Vec::new();
    for i in 0..5 {
        let sensor = tx
            .create_entity()
            .expect("failed")
            .with_label("Sensor")
            .with_property("type", "temperature")
            .with_property("zone", format!("zone_{}", i))
            .with_property("calibration_offset", 0.0f64);
        sensor_ids.push(sensor.id);
        tx.put_entity(&sensor).expect("failed");

        // Connect sensor to gateway
        let edge = tx.create_edge(gateway.id, sensor.id, "MONITORS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Operational phase: Record sensor readings
    let mut reading_ids = Vec::new();
    for cycle in 0..10 {
        let mut tx = db.begin().expect("failed to begin");

        for (i, &sensor_id) in sensor_ids.iter().enumerate() {
            // Simulate temperature reading with some variation
            let temp = 20.0 + (i as f32 * 0.5) + (cycle as f32 * 0.1);

            // Create reading entity with embedding for anomaly detection
            let embedding = vec![
                temp / 50.0,
                (cycle as f32) / 20.0,
                (i as f32) / 10.0,
                0.5,
            ];

            let reading = tx
                .create_entity()
                .expect("failed")
                .with_label("Reading")
                .with_property("temperature", temp as f64)
                .with_property("timestamp", (1000 + cycle * 60) as i64)
                .with_property("embedding", embedding);
            reading_ids.push(reading.id);
            tx.put_entity(&reading).expect("failed");

            // Link reading to sensor
            let edge = tx.create_edge(sensor_id, reading.id, "RECORDED").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed");
    }

    // Query phase: Verify topology
    let tx = db.begin_read().expect("failed");

    // Verify gateway monitors all sensors
    let monitored = tx.get_outgoing_edges(gateway.id).expect("failed");
    let monitored_sensors: Vec<_> = monitored
        .iter()
        .filter(|e| e.edge_type.as_str() == "MONITORS")
        .collect();
    assert_eq!(monitored_sensors.len(), 5);

    // Verify each sensor has readings
    for &sensor_id in &sensor_ids {
        let readings = tx.get_outgoing_edges(sensor_id).expect("failed");
        let recorded: Vec<_> = readings
            .iter()
            .filter(|e| e.edge_type.as_str() == "RECORDED")
            .collect();
        assert_eq!(recorded.len(), 10);
    }

    // Find hottest reading across all sensors
    let mut max_temp = f64::MIN;
    let mut hottest_reading = None;
    for &reading_id in &reading_ids {
        let reading = tx.get_entity(reading_id).expect("failed").expect("not found");
        if let Some(Value::Float(temp)) = reading.get_property("temperature") {
            if *temp > max_temp {
                max_temp = *temp;
                hottest_reading = Some(reading_id);
            }
        }
    }
    assert!(hottest_reading.is_some());
    assert!(max_temp > 20.0);
}

// ============================================================================
// Local-First Notes Application
// ============================================================================

/// Simulates a local-first notes app with:
/// - Graph: Folder hierarchy, note links, tags
/// - Vector: Semantic embeddings for similar note search
/// - Properties: Content, timestamps, metadata
#[test]
fn test_local_notes_application() {
    let db = Database::in_memory().expect("failed to create db");

    // Create folder structure
    let mut tx = db.begin().expect("failed to begin");

    let root = tx
        .create_entity()
        .expect("failed")
        .with_label("Folder")
        .with_property("name", "Notes")
        .with_property("is_root", true);
    tx.put_entity(&root).expect("failed");

    let work = tx
        .create_entity()
        .expect("failed")
        .with_label("Folder")
        .with_property("name", "Work");
    tx.put_entity(&work).expect("failed");

    let personal = tx
        .create_entity()
        .expect("failed")
        .with_label("Folder")
        .with_property("name", "Personal");
    tx.put_entity(&personal).expect("failed");

    // Folder hierarchy
    let e1 = tx.create_edge(root.id, work.id, "CONTAINS").expect("failed");
    let e2 = tx.create_edge(root.id, personal.id, "CONTAINS").expect("failed");
    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");

    // Create tags
    let mut tag_ids = HashMap::new();
    for tag_name in ["important", "todo", "idea", "reference"] {
        let tag = tx
            .create_entity()
            .expect("failed")
            .with_label("Tag")
            .with_property("name", tag_name);
        tag_ids.insert(tag_name.to_string(), tag.id);
        tx.put_entity(&tag).expect("failed");
    }

    tx.commit().expect("failed");

    // Create notes with embeddings
    let notes_data = vec![
        ("Meeting Notes", work.id, vec!["important", "todo"], vec![0.8f32, 0.2, 0.1, 0.3]),
        ("Project Ideas", work.id, vec!["idea"], vec![0.7f32, 0.3, 0.4, 0.2]),
        ("Vacation Plans", personal.id, vec!["todo"], vec![0.1f32, 0.8, 0.7, 0.1]),
        ("Reading List", personal.id, vec!["reference"], vec![0.2f32, 0.7, 0.3, 0.5]),
        ("Sprint Review", work.id, vec!["important", "reference"], vec![0.75f32, 0.25, 0.15, 0.35]),
    ];

    let mut note_ids = Vec::new();
    {
        let mut tx = db.begin().expect("failed to begin");

        for (title, folder_id, tags, embedding) in notes_data {
            let note = tx
                .create_entity()
                .expect("failed")
                .with_label("Note")
                .with_property("title", title)
                .with_property("content", format!("Content of {}", title))
                .with_property("created_at", 1700000000i64)
                .with_property("embedding", embedding);
            note_ids.push(note.id);
            tx.put_entity(&note).expect("failed");

            // Place in folder
            let folder_edge = tx.create_edge(folder_id, note.id, "CONTAINS").expect("failed");
            tx.put_edge(&folder_edge).expect("failed");

            // Apply tags
            for tag_name in tags {
                let tag_edge = tx
                    .create_edge(note.id, *tag_ids.get(tag_name).unwrap(), "TAGGED")
                    .expect("failed");
                tx.put_edge(&tag_edge).expect("failed");
            }
        }

        // Create links between notes
        let link = tx.create_edge(note_ids[0], note_ids[4], "LINKS_TO").expect("failed");
        tx.put_edge(&link).expect("failed");

        tx.commit().expect("failed");
    }

    // Query: Find all important notes
    let tx = db.begin_read().expect("failed");
    let important_tag = tag_ids.get("important").unwrap();
    let tagged_edges = tx.get_incoming_edges(*important_tag).expect("failed");
    let important_notes: Vec<_> = tagged_edges
        .iter()
        .filter(|e| e.edge_type.as_str() == "TAGGED")
        .map(|e| e.source)
        .collect();
    assert_eq!(important_notes.len(), 2);

    // Query: Find notes in work folder
    let work_contents = tx.get_outgoing_edges(work.id).expect("failed");
    let work_notes: Vec<_> = work_contents
        .iter()
        .filter(|e| e.edge_type.as_str() == "CONTAINS")
        .collect();
    assert_eq!(work_notes.len(), 3);

    // Query: Find similar notes using embeddings
    let query_embedding = vec![0.78f32, 0.22, 0.12, 0.32]; // Similar to work notes

    let mut similarities: Vec<(EntityId, f32)> = Vec::new();
    for &note_id in &note_ids {
        let note = tx.get_entity(note_id).expect("failed").expect("not found");
        if let Some(Value::Vector(emb)) = note.get_property("embedding") {
            let dot: f32 = query_embedding.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            similarities.push((note_id, dot));
        }
    }
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Most similar should be work-related notes
    let top_note = tx.get_entity(similarities[0].0).expect("failed").expect("not found");
    let title = top_note.get_property("title");
    assert!(
        title == Some(&Value::String("Meeting Notes".to_string()))
            || title == Some(&Value::String("Sprint Review".to_string()))
    );
}

// ============================================================================
// Embedded Recommendation Engine
// ============================================================================

/// Simulates a recommendation engine for a media player:
/// - Graph: User preferences, artist relationships
/// - Vector: Song embeddings for similarity
/// - Properties: Play counts, ratings, metadata
#[test]
fn test_recommendation_engine() {
    let db = Database::in_memory().expect("failed to create db");

    // Create music library
    let mut tx = db.begin().expect("failed to begin");

    // Artists
    let artists = vec![
        ("Artist A", "rock"),
        ("Artist B", "pop"),
        ("Artist C", "rock"),
        ("Artist D", "electronic"),
    ];

    let mut artist_ids = Vec::new();
    for (name, genre) in &artists {
        let artist = tx
            .create_entity()
            .expect("failed")
            .with_label("Artist")
            .with_property("name", *name)
            .with_property("genre", *genre);
        artist_ids.push(artist.id);
        tx.put_entity(&artist).expect("failed");
    }

    // Songs with embeddings representing audio features
    let songs = vec![
        ("Rock Song 1", artist_ids[0], vec![0.9f32, 0.1, 0.8, 0.2]),
        ("Rock Song 2", artist_ids[0], vec![0.85f32, 0.15, 0.75, 0.25]),
        ("Pop Hit", artist_ids[1], vec![0.3f32, 0.9, 0.4, 0.5]),
        ("Rock Ballad", artist_ids[2], vec![0.7f32, 0.3, 0.6, 0.4]),
        ("EDM Track", artist_ids[3], vec![0.2f32, 0.5, 0.3, 0.9]),
    ];

    let mut song_ids = Vec::new();
    for (title, artist_id, embedding) in songs {
        let song = tx
            .create_entity()
            .expect("failed")
            .with_label("Song")
            .with_property("title", title)
            .with_property("embedding", embedding)
            .with_property("duration", 180i64);
        song_ids.push(song.id);
        tx.put_entity(&song).expect("failed");

        let edge = tx.create_edge(artist_id, song.id, "PERFORMS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    // Create user with play history
    let user = tx
        .create_entity()
        .expect("failed")
        .with_label("User")
        .with_property("username", "music_fan");
    tx.put_entity(&user).expect("failed");

    // User's play history (heavily favoring rock)
    let plays = vec![
        (song_ids[0], 50), // Rock Song 1 - played 50 times
        (song_ids[1], 30), // Rock Song 2 - played 30 times
        (song_ids[3], 20), // Rock Ballad - played 20 times
        (song_ids[2], 2),  // Pop Hit - played 2 times
    ];

    for (song_id, play_count) in plays {
        let play_edge = tx
            .create_edge(user.id, song_id, "PLAYED")
            .expect("failed")
            .with_property("count", play_count as i64);
        tx.put_edge(&play_edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Recommendation query: Find songs similar to user's favorites
    let tx = db.begin_read().expect("failed");

    // Get user's most played songs
    let user_plays = tx.get_outgoing_edges(user.id).expect("failed");
    let mut favorites: Vec<(EntityId, i64)> = user_plays
        .iter()
        .filter(|e| e.edge_type.as_str() == "PLAYED")
        .filter_map(|e| {
            e.get_property("count")
                .and_then(|v| if let Value::Int(c) = v { Some((e.target, *c)) } else { None })
        })
        .collect();
    favorites.sort_by(|a, b| b.1.cmp(&a.1));

    // Get embedding of top favorite
    let top_song = tx.get_entity(favorites[0].0).expect("failed").expect("not found");
    let top_embedding = match top_song.get_property("embedding") {
        Some(Value::Vector(v)) => v.clone(),
        _ => panic!("missing embedding"),
    };

    // Find similar songs user hasn't played much
    let played_a_lot: HashSet<_> = favorites.iter().filter(|(_, c)| *c > 10).map(|(id, _)| *id).collect();

    let mut recommendations = Vec::new();
    for &song_id in &song_ids {
        if played_a_lot.contains(&song_id) {
            continue;
        }

        let song = tx.get_entity(song_id).expect("failed").expect("not found");
        if let Some(Value::Vector(emb)) = song.get_property("embedding") {
            let similarity: f32 = top_embedding.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            recommendations.push((song_id, similarity));
        }
    }
    recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // EDM track should not be top recommendation (different genre)
    assert_ne!(recommendations[0].0, song_ids[4]);
}

// ============================================================================
// Offline-First Task Manager
// ============================================================================

/// Simulates a task manager with:
/// - Graph: Project hierarchy, task dependencies
/// - Properties: Status, priority, due dates
#[test]
fn test_task_manager() {
    let db = Database::in_memory().expect("failed to create db");

    // Create workspace
    let mut tx = db.begin().expect("failed to begin");

    let workspace = tx
        .create_entity()
        .expect("failed")
        .with_label("Workspace")
        .with_property("name", "My Tasks");
    tx.put_entity(&workspace).expect("failed");

    // Create projects
    let project1 = tx
        .create_entity()
        .expect("failed")
        .with_label("Project")
        .with_property("name", "Website Redesign")
        .with_property("status", "active");
    tx.put_entity(&project1).expect("failed");

    let project2 = tx
        .create_entity()
        .expect("failed")
        .with_label("Project")
        .with_property("name", "Mobile App")
        .with_property("status", "planning");
    tx.put_entity(&project2).expect("failed");

    // Workspace contains projects
    let e1 = tx.create_edge(workspace.id, project1.id, "CONTAINS").expect("failed");
    let e2 = tx.create_edge(workspace.id, project2.id, "CONTAINS").expect("failed");
    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");

    // Create tasks for project 1
    let tasks = vec![
        ("Design mockups", "done", 1),
        ("Implement frontend", "in_progress", 2),
        ("Write tests", "todo", 3),
        ("Deploy", "todo", 4),
    ];

    let mut task_ids = Vec::new();
    for (title, status, priority) in tasks {
        let task = tx
            .create_entity()
            .expect("failed")
            .with_label("Task")
            .with_property("title", title)
            .with_property("status", status)
            .with_property("priority", priority as i64);
        task_ids.push(task.id);
        tx.put_entity(&task).expect("failed");

        let edge = tx.create_edge(project1.id, task.id, "HAS_TASK").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    // Task dependencies
    // Implement frontend depends on Design mockups
    let dep1 = tx.create_edge(task_ids[1], task_ids[0], "DEPENDS_ON").expect("failed");
    tx.put_edge(&dep1).expect("failed");

    // Write tests depends on Implement frontend
    let dep2 = tx.create_edge(task_ids[2], task_ids[1], "DEPENDS_ON").expect("failed");
    tx.put_edge(&dep2).expect("failed");

    // Deploy depends on Write tests
    let dep3 = tx.create_edge(task_ids[3], task_ids[2], "DEPENDS_ON").expect("failed");
    tx.put_edge(&dep3).expect("failed");

    tx.commit().expect("failed");

    // Query: Find blocked tasks (dependencies not done)
    let tx = db.begin_read().expect("failed");

    let mut blocked_tasks = Vec::new();
    for &task_id in &task_ids {
        let task = tx.get_entity(task_id).expect("failed").expect("not found");
        if task.get_property("status") == Some(&Value::String("todo".to_string())) {
            // Check if all dependencies are done
            let deps = tx.get_outgoing_edges(task_id).expect("failed");
            let mut all_deps_done = true;

            for dep in deps.iter().filter(|e| e.edge_type.as_str() == "DEPENDS_ON") {
                let dep_task = tx.get_entity(dep.target).expect("failed").expect("not found");
                if dep_task.get_property("status") != Some(&Value::String("done".to_string())) {
                    all_deps_done = false;
                    break;
                }
            }

            if !all_deps_done {
                blocked_tasks.push(task_id);
            }
        }
    }

    // "Write tests" is blocked (depends on in_progress task)
    // "Deploy" is blocked (depends on todo task)
    assert_eq!(blocked_tasks.len(), 2);

    // Query: Find next actionable task (todo with all deps done)
    let mut actionable = Vec::new();
    for &task_id in &task_ids {
        let task = tx.get_entity(task_id).expect("failed").expect("not found");
        if task.get_property("status") != Some(&Value::String("todo".to_string())) {
            continue;
        }

        let deps = tx.get_outgoing_edges(task_id).expect("failed");
        let all_deps_done = deps
            .iter()
            .filter(|e| e.edge_type.as_str() == "DEPENDS_ON")
            .all(|e| {
                let dep = tx.get_entity(e.target).expect("failed").expect("not found");
                dep.get_property("status") == Some(&Value::String("done".to_string()))
            });

        if all_deps_done {
            actionable.push(task_id);
        }
    }

    // No actionable tasks (all todos are blocked)
    assert_eq!(actionable.len(), 0);

    // Complete the in-progress task and check again
    drop(tx);

    {
        let mut tx = db.begin().expect("failed");
        let mut task = tx.get_entity(task_ids[1]).expect("failed").expect("not found");
        task.set_property("status", "done");
        tx.put_entity(&task).expect("failed");
        tx.commit().expect("failed");
    }

    // Now "Write tests" should be actionable
    let tx = db.begin_read().expect("failed");
    let write_tests = tx.get_entity(task_ids[2]).expect("failed").expect("not found");
    let deps = tx.get_outgoing_edges(task_ids[2]).expect("failed");
    let all_deps_done = deps
        .iter()
        .filter(|e| e.edge_type.as_str() == "DEPENDS_ON")
        .all(|e| {
            let dep = tx.get_entity(e.target).expect("failed").expect("not found");
            dep.get_property("status") == Some(&Value::String("done".to_string()))
        });
    assert!(all_deps_done);
    assert_eq!(
        write_tests.get_property("status"),
        Some(&Value::String("todo".to_string()))
    );
}

// ============================================================================
// Multi-Transaction Workflow
// ============================================================================

/// Tests a realistic workflow with multiple sequential transactions
#[test]
fn test_multi_transaction_workflow() {
    let db = Database::in_memory().expect("failed to create db");

    // Transaction 1: Initial setup
    let user_id;
    {
        let mut tx = db.begin().expect("failed");
        let user = tx
            .create_entity()
            .expect("failed")
            .with_label("User")
            .with_property("email", "test@example.com")
            .with_property("credits", 100i64);
        user_id = user.id;
        tx.put_entity(&user).expect("failed");
        tx.commit().expect("failed");
    }

    // Transaction 2: User makes a purchase
    let order_id;
    {
        let mut tx = db.begin().expect("failed");

        // Verify user has enough credits
        let user = tx.get_entity(user_id).expect("failed").expect("not found");
        let credits = match user.get_property("credits") {
            Some(Value::Int(c)) => *c,
            _ => panic!("missing credits"),
        };
        assert!(credits >= 50);

        // Create order
        let order = tx
            .create_entity()
            .expect("failed")
            .with_label("Order")
            .with_property("amount", 50i64)
            .with_property("status", "pending");
        order_id = order.id;
        tx.put_entity(&order).expect("failed");

        // Link to user
        let edge = tx.create_edge(user_id, order_id, "PLACED").expect("failed");
        tx.put_edge(&edge).expect("failed");

        // Deduct credits
        let mut user = tx.get_entity(user_id).expect("failed").expect("not found");
        user.set_property("credits", credits - 50);
        tx.put_entity(&user).expect("failed");

        tx.commit().expect("failed");
    }

    // Transaction 3: Mark order as completed
    {
        let mut tx = db.begin().expect("failed");
        let mut order = tx.get_entity(order_id).expect("failed").expect("not found");
        order.set_property("status", "completed");
        tx.put_entity(&order).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify final state
    let tx = db.begin_read().expect("failed");

    let user = tx.get_entity(user_id).expect("failed").expect("not found");
    assert_eq!(user.get_property("credits"), Some(&Value::Int(50)));

    let order = tx.get_entity(order_id).expect("failed").expect("not found");
    assert_eq!(
        order.get_property("status"),
        Some(&Value::String("completed".to_string()))
    );

    let user_orders = tx.get_outgoing_edges(user_id).expect("failed");
    assert_eq!(user_orders.len(), 1);
    assert_eq!(user_orders[0].target, order_id);
}

// ============================================================================
// Concurrent Read Operations
// ============================================================================

/// Tests multiple concurrent read transactions
#[test]
fn test_concurrent_reads_workflow() {
    let db = Database::in_memory().expect("failed to create db");

    // Setup data
    {
        let mut tx = db.begin().expect("failed");
        for i in 0..100 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("Item")
                .with_property("index", i as i64)
                .with_property("value", (i * i) as i64);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Multiple concurrent reads
    let tx1 = db.begin_read().expect("failed");
    let tx2 = db.begin_read().expect("failed");

    // Both should see the same data
    for i in 1..=100 {
        let id = EntityId::new(i);
        let e1 = tx1.get_entity(id).expect("failed").expect("not found");
        let e2 = tx2.get_entity(id).expect("failed").expect("not found");

        assert_eq!(e1.get_property("index"), e2.get_property("index"));
        assert_eq!(e1.get_property("value"), e2.get_property("value"));
    }
}

// ============================================================================
// Data Integrity After Updates
// ============================================================================

/// Tests that data integrity is maintained through updates
#[test]
fn test_update_integrity() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities with relationships
    let parent_id;
    let mut child_ids = Vec::new();
    {
        let mut tx = db.begin().expect("failed");

        let parent = tx
            .create_entity()
            .expect("failed")
            .with_label("Parent")
            .with_property("name", "Root");
        parent_id = parent.id;
        tx.put_entity(&parent).expect("failed");

        for i in 0..5 {
            let child = tx
                .create_entity()
                .expect("failed")
                .with_label("Child")
                .with_property("name", format!("Child {}", i))
                .with_property("order", i as i64);
            child_ids.push(child.id);
            tx.put_entity(&child).expect("failed");

            let edge = tx.create_edge(parent_id, child.id, "HAS_CHILD").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed");
    }

    // Update parent
    {
        let mut tx = db.begin().expect("failed");
        let mut parent = tx.get_entity(parent_id).expect("failed").expect("not found");
        parent.set_property("name", "Updated Root");
        parent.set_property("updated", true);
        tx.put_entity(&parent).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify children still connected
    let tx = db.begin_read().expect("failed");

    let children = tx.get_outgoing_edges(parent_id).expect("failed");
    assert_eq!(children.len(), 5);

    for &child_id in &child_ids {
        let child = tx.get_entity(child_id).expect("failed").expect("not found");
        assert!(child.get_property("name").is_some());
    }

    let parent = tx.get_entity(parent_id).expect("failed").expect("not found");
    assert_eq!(
        parent.get_property("name"),
        Some(&Value::String("Updated Root".to_string()))
    );
    assert_eq!(parent.get_property("updated"), Some(&Value::Bool(true)));
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Tests efficient batch operations
#[test]
fn test_batch_operations() {
    let db = Database::in_memory().expect("failed to create db");

    // Batch insert
    let mut entity_ids = Vec::new();
    {
        let mut tx = db.begin().expect("failed");

        for i in 0..1000 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("BatchItem")
                .with_property("index", i as i64)
                .with_property("batch", "initial");
            entity_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        tx.commit().expect("failed");
    }

    // Batch update
    {
        let mut tx = db.begin().expect("failed");

        for &id in entity_ids.iter().take(500) {
            let mut entity = tx.get_entity(id).expect("failed").expect("not found");
            entity.set_property("batch", "updated");
            tx.put_entity(&entity).expect("failed");
        }

        tx.commit().expect("failed");
    }

    // Verify
    let tx = db.begin_read().expect("failed");

    let mut updated_count = 0;
    let mut initial_count = 0;

    for &id in &entity_ids {
        let entity = tx.get_entity(id).expect("failed").expect("not found");
        match entity.get_property("batch") {
            Some(Value::String(s)) if s == "updated" => updated_count += 1,
            Some(Value::String(s)) if s == "initial" => initial_count += 1,
            _ => panic!("unexpected batch value"),
        }
    }

    assert_eq!(updated_count, 500);
    assert_eq!(initial_count, 500);
}
