//! Integration tests for NodeStore.

use manifoldb_core::{Entity, EntityId, Label, Value};
use manifoldb_graph::store::{GraphError, IdGenerator, NodeStore};
use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{StorageEngine, Transaction};

fn create_test_engine() -> RedbEngine {
    RedbEngine::in_memory().expect("Failed to create in-memory engine")
}

#[test]
fn create_and_get_entity() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity = NodeStore::create(&mut tx, &id_gen, |id| {
        Entity::new(id).with_label("Person").with_property("name", "Alice")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = NodeStore::get(&tx, entity.id).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, entity.id);
    assert!(retrieved.has_label("Person"));
    assert_eq!(retrieved.get_property("name"), Some(&Value::String("Alice".to_owned())));
}

#[test]
fn create_with_id() {
    let engine = create_test_engine();

    let entity = Entity::new(EntityId::new(42)).with_label("Test").with_property("key", "value");

    let mut tx = engine.begin_write().unwrap();
    NodeStore::create_with_id(&mut tx, &entity).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = NodeStore::get(&tx, EntityId::new(42)).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id.as_u64(), 42);
}

#[test]
fn create_with_id_duplicate_fails() {
    let engine = create_test_engine();

    let entity = Entity::new(EntityId::new(1)).with_label("Test");

    let mut tx = engine.begin_write().unwrap();
    NodeStore::create_with_id(&mut tx, &entity).unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let result = NodeStore::create_with_id(&mut tx, &entity);
    assert!(matches!(result, Err(GraphError::EntityAlreadyExists(_))));
}

#[test]
fn get_nonexistent_returns_none() {
    let engine = create_test_engine();

    let tx = engine.begin_read().unwrap();
    let result = NodeStore::get(&tx, EntityId::new(999)).unwrap();
    assert!(result.is_none());
}

#[test]
fn get_or_error_nonexistent_returns_error() {
    let engine = create_test_engine();

    let tx = engine.begin_read().unwrap();
    let result = NodeStore::get_or_error(&tx, EntityId::new(999));
    assert!(matches!(result, Err(GraphError::EntityNotFound(_))));
}

#[test]
fn exists_check() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert!(NodeStore::exists(&tx, entity.id).unwrap());
    assert!(!NodeStore::exists(&tx, EntityId::new(999)).unwrap());
}

#[test]
fn update_entity() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_property("count", 1i64))
            .unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let mut updated = entity.clone();
    updated.set_property("count", 2i64);
    updated.labels.push(Label::new("Updated"));
    NodeStore::update(&mut tx, &updated).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = NodeStore::get(&tx, entity.id).unwrap().unwrap();
    assert_eq!(retrieved.get_property("count"), Some(&Value::Int(2)));
    assert!(retrieved.has_label("Updated"));
}

#[test]
fn update_nonexistent_fails() {
    let engine = create_test_engine();

    let entity = Entity::new(EntityId::new(999));

    let mut tx = engine.begin_write().unwrap();
    let result = NodeStore::update(&mut tx, &entity);
    assert!(matches!(result, Err(GraphError::EntityNotFound(_))));
}

#[test]
fn delete_entity() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("ToDelete")).unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    assert!(NodeStore::delete(&mut tx, entity.id).unwrap());
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert!(!NodeStore::exists(&tx, entity.id).unwrap());
}

#[test]
fn delete_nonexistent_returns_false() {
    let engine = create_test_engine();

    let mut tx = engine.begin_write().unwrap();
    assert!(!NodeStore::delete(&mut tx, EntityId::new(999)).unwrap());
}

#[test]
fn find_by_label() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let person1 =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Person")).unwrap();
    let person2 =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Person")).unwrap();
    let _company =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("Company")).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let persons = NodeStore::find_by_label(&tx, &Label::new("Person")).unwrap();
    assert_eq!(persons.len(), 2);
    assert!(persons.contains(&person1.id));
    assert!(persons.contains(&person2.id));

    let companies = NodeStore::find_by_label(&tx, &Label::new("Company")).unwrap();
    assert_eq!(companies.len(), 1);

    let nonexistent = NodeStore::find_by_label(&tx, &Label::new("NonExistent")).unwrap();
    assert!(nonexistent.is_empty());
}

#[test]
fn count_entities() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let tx = engine.begin_read().unwrap();
    assert_eq!(NodeStore::count(&tx).unwrap(), 0);
    drop(tx);

    let mut tx = engine.begin_write().unwrap();
    for _ in 0..5 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(NodeStore::count(&tx).unwrap(), 5);
}

#[test]
fn for_each_iteration() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    for i in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_property("index", i as i64))
            .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let mut count = 0;
    NodeStore::for_each(&tx, |_entity| {
        count += 1;
        true
    })
    .unwrap();
    assert_eq!(count, 10);
}

#[test]
fn for_each_early_termination() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    for _ in 0..10 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let mut count = 0;
    NodeStore::for_each(&tx, |_entity| {
        count += 1;
        count < 5
    })
    .unwrap();
    assert_eq!(count, 5);
}

#[test]
fn all_entities() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    for _ in 0..5 {
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let all = NodeStore::all(&tx).unwrap();
    assert_eq!(all.len(), 5);
}

#[test]
fn label_index_updated_on_update() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create entity with one label
    let mut tx = engine.begin_write().unwrap();
    let entity =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("OldLabel")).unwrap();
    tx.commit().unwrap();

    // Update with different label
    let mut tx = engine.begin_write().unwrap();
    let mut updated = entity.clone();
    updated.labels.clear();
    updated.labels.push(Label::new("NewLabel"));
    NodeStore::update(&mut tx, &updated).unwrap();
    tx.commit().unwrap();

    // Verify old label no longer finds the entity
    let tx = engine.begin_read().unwrap();
    let old_results = NodeStore::find_by_label(&tx, &Label::new("OldLabel")).unwrap();
    assert!(old_results.is_empty());

    // Verify new label finds the entity
    let new_results = NodeStore::find_by_label(&tx, &Label::new("NewLabel")).unwrap();
    assert_eq!(new_results.len(), 1);
    assert_eq!(new_results[0], entity.id);
}

#[test]
fn label_index_cleaned_up_on_delete() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_label("ToDelete")).unwrap();
    tx.commit().unwrap();

    // Verify label index before delete
    let tx = engine.begin_read().unwrap();
    let results = NodeStore::find_by_label(&tx, &Label::new("ToDelete")).unwrap();
    assert_eq!(results.len(), 1);
    drop(tx);

    // Delete entity
    let mut tx = engine.begin_write().unwrap();
    NodeStore::delete(&mut tx, entity.id).unwrap();
    tx.commit().unwrap();

    // Verify label index is cleaned up
    let tx = engine.begin_read().unwrap();
    let results = NodeStore::find_by_label(&tx, &Label::new("ToDelete")).unwrap();
    assert!(results.is_empty());
}

#[test]
fn entity_with_multiple_labels() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity = NodeStore::create(&mut tx, &id_gen, |id| {
        Entity::new(id).with_label("Person").with_label("Employee").with_label("Manager")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Should be findable by all labels
    let persons = NodeStore::find_by_label(&tx, &Label::new("Person")).unwrap();
    assert!(persons.contains(&entity.id));

    let employees = NodeStore::find_by_label(&tx, &Label::new("Employee")).unwrap();
    assert!(employees.contains(&entity.id));

    let managers = NodeStore::find_by_label(&tx, &Label::new("Manager")).unwrap();
    assert!(managers.contains(&entity.id));
}

#[test]
fn id_generator_produces_unique_ids() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let mut ids = Vec::new();
    for _ in 0..100 {
        let entity = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        ids.push(entity.id);
    }
    tx.commit().unwrap();

    // All IDs should be unique
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 100);
}

#[test]
fn max_id_returns_highest() {
    let engine = create_test_engine();

    // Empty store returns None
    let tx = engine.begin_read().unwrap();
    assert!(NodeStore::max_id(&tx).unwrap().is_none());
    drop(tx);

    // Create entities with specific IDs
    let mut tx = engine.begin_write().unwrap();
    NodeStore::create_with_id(&mut tx, &Entity::new(EntityId::new(10))).unwrap();
    NodeStore::create_with_id(&mut tx, &Entity::new(EntityId::new(5))).unwrap();
    NodeStore::create_with_id(&mut tx, &Entity::new(EntityId::new(100))).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let max = NodeStore::max_id(&tx).unwrap();
    assert_eq!(max, Some(EntityId::new(100)));
}

#[test]
fn entity_with_various_property_types() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let entity = NodeStore::create(&mut tx, &id_gen, |id| {
        Entity::new(id)
            .with_property("string", "hello")
            .with_property("int", 42i64)
            .with_property("float", 2.71f64)
            .with_property("bool", true)
            .with_property("vector", Value::Vector(vec![1.0, 2.0, 3.0]))
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = NodeStore::get(&tx, entity.id).unwrap().unwrap();
    assert_eq!(retrieved.get_property("string"), Some(&Value::String("hello".to_owned())));
    assert_eq!(retrieved.get_property("int"), Some(&Value::Int(42)));
    assert_eq!(retrieved.get_property("float"), Some(&Value::Float(2.71)));
    assert_eq!(retrieved.get_property("bool"), Some(&Value::Bool(true)));
    assert_eq!(retrieved.get_property("vector"), Some(&Value::Vector(vec![1.0, 2.0, 3.0])));
}
