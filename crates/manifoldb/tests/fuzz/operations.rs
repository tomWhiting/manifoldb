//! Random operation generation for fuzz testing.
//!
//! Generates random sequences of database operations and verifies
//! they don't cause panics or inconsistencies.

use proptest::prelude::*;
use std::collections::HashSet;

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Operation Types
// ============================================================================

/// A database operation that can be randomly generated and applied
#[derive(Debug, Clone)]
pub enum Operation {
    /// Create a new entity
    CreateEntity { label: Option<String>, properties: Vec<(String, TestValue)> },
    /// Update an existing entity
    UpdateEntity { entity_idx: usize, properties: Vec<(String, TestValue)> },
    /// Delete an entity
    DeleteEntity { entity_idx: usize },
    /// Create an edge between entities
    CreateEdge { source_idx: usize, target_idx: usize, edge_type: String },
    /// Delete an edge
    DeleteEdge { edge_idx: usize },
    /// Read an entity
    ReadEntity { entity_idx: usize },
    /// Get outgoing edges
    GetOutgoingEdges { entity_idx: usize },
    /// Get incoming edges
    GetIncomingEdges { entity_idx: usize },
    /// Commit current transaction
    Commit,
    /// Rollback current transaction
    Rollback,
}

/// Test value types that can be randomly generated
#[derive(Debug, Clone)]
pub enum TestValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vector(Vec<f32>),
}

impl From<TestValue> for Value {
    fn from(tv: TestValue) -> Self {
        match tv {
            TestValue::Null => Value::Null,
            TestValue::Bool(b) => Value::Bool(b),
            TestValue::Int(i) => Value::Int(i),
            TestValue::Float(f) => Value::Float(f),
            TestValue::String(s) => Value::String(s),
            TestValue::Vector(v) => Value::Vector(v),
        }
    }
}

// ============================================================================
// Proptest Strategies
// ============================================================================

/// Strategy for generating test values
fn test_value_strategy() -> impl Strategy<Value = TestValue> {
    prop_oneof![
        Just(TestValue::Null),
        any::<bool>().prop_map(TestValue::Bool),
        any::<i64>().prop_map(TestValue::Int),
        any::<f64>().prop_filter("finite", |f| f.is_finite()).prop_map(TestValue::Float),
        "[a-z]{1,20}".prop_map(TestValue::String),
        proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..32)
            .prop_map(TestValue::Vector),
    ]
}

/// Strategy for generating property key-value pairs
fn properties_strategy() -> impl Strategy<Value = Vec<(String, TestValue)>> {
    proptest::collection::vec(("[a-z_]{1,10}".prop_map(String::from), test_value_strategy()), 0..5)
}

/// Strategy for generating labels
fn label_strategy() -> impl Strategy<Value = Option<String>> {
    prop_oneof![Just(None), "[A-Z][a-z]{0,10}".prop_map(Some),]
}

/// Strategy for generating edge types
fn edge_type_strategy() -> impl Strategy<Value = String> {
    "[A-Z_]{1,10}".prop_map(String::from)
}

/// Strategy for generating a single operation
fn operation_strategy(max_entities: usize, max_edges: usize) -> impl Strategy<Value = Operation> {
    let entity_idx = 0..max_entities.max(1);
    let edge_idx = 0..max_edges.max(1);

    prop_oneof![
        // CreateEntity
        (label_strategy(), properties_strategy())
            .prop_map(|(label, properties)| Operation::CreateEntity { label, properties }),
        // UpdateEntity
        (entity_idx.clone(), properties_strategy()).prop_map(|(entity_idx, properties)| {
            Operation::UpdateEntity { entity_idx, properties }
        }),
        // DeleteEntity
        entity_idx.clone().prop_map(|entity_idx| Operation::DeleteEntity { entity_idx }),
        // CreateEdge
        (entity_idx.clone(), entity_idx.clone(), edge_type_strategy()).prop_map(
            |(source_idx, target_idx, edge_type)| Operation::CreateEdge {
                source_idx,
                target_idx,
                edge_type
            }
        ),
        // DeleteEdge
        edge_idx.prop_map(|edge_idx| Operation::DeleteEdge { edge_idx }),
        // ReadEntity
        entity_idx.clone().prop_map(|entity_idx| Operation::ReadEntity { entity_idx }),
        // GetOutgoingEdges
        entity_idx.clone().prop_map(|entity_idx| Operation::GetOutgoingEdges { entity_idx }),
        // GetIncomingEdges
        entity_idx.prop_map(|entity_idx| Operation::GetIncomingEdges { entity_idx }),
        // Transaction control
        Just(Operation::Commit),
        Just(Operation::Rollback),
    ]
}

/// Strategy for generating a sequence of operations
pub fn operations_strategy(count: usize) -> impl Strategy<Value = Vec<Operation>> {
    proptest::collection::vec(operation_strategy(100, 200), count)
}

// ============================================================================
// Operation Executor
// ============================================================================

/// Tracks state during operation execution
#[allow(dead_code)]
pub struct ExecutionState {
    /// Created entity IDs
    pub entity_ids: Vec<EntityId>,
    /// Created edge IDs (source, target, edge_id)
    pub edge_info: Vec<(EntityId, EntityId, manifoldb::EdgeId)>,
    /// Whether we're in a transaction (for future multi-op transaction support)
    pub in_transaction: bool,
    /// Uncommitted entity IDs (for future rollback tracking)
    pub uncommitted_entities: HashSet<EntityId>,
    /// Uncommitted edge IDs (for future rollback tracking)
    pub uncommitted_edges: HashSet<manifoldb::EdgeId>,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self {
            entity_ids: Vec::new(),
            edge_info: Vec::new(),
            in_transaction: false,
            uncommitted_entities: HashSet::new(),
            uncommitted_edges: HashSet::new(),
        }
    }

    pub fn get_entity_id(&self, idx: usize) -> Option<EntityId> {
        if self.entity_ids.is_empty() {
            None
        } else {
            Some(self.entity_ids[idx % self.entity_ids.len()])
        }
    }

    pub fn get_edge_info(&self, idx: usize) -> Option<(EntityId, EntityId, manifoldb::EdgeId)> {
        if self.edge_info.is_empty() {
            None
        } else {
            Some(self.edge_info[idx % self.edge_info.len()])
        }
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a sequence of operations, tracking panics
pub fn execute_operations(
    db: &Database,
    operations: &[Operation],
) -> Result<ExecutionState, String> {
    let mut state = ExecutionState::new();

    for (i, op) in operations.iter().enumerate() {
        if let Err(e) = execute_single_operation(db, op, &mut state) {
            // Some errors are expected (e.g., operating on non-existent entities)
            // We just continue
            eprintln!("Operation {i} ({op:?}) failed: {e}");
        }
    }

    Ok(state)
}

fn execute_single_operation(
    db: &Database,
    op: &Operation,
    state: &mut ExecutionState,
) -> Result<(), String> {
    match op {
        Operation::CreateEntity { label, properties } => {
            let mut tx = db.begin().map_err(|e| e.to_string())?;
            let mut entity = tx.create_entity().map_err(|e| e.to_string())?;

            if let Some(l) = label {
                entity = entity.with_label(l.as_str());
            }

            for (key, value) in properties {
                entity.set_property(key, Value::from(value.clone()));
            }

            let id = entity.id;
            tx.put_entity(&entity).map_err(|e| e.to_string())?;
            tx.commit().map_err(|e| e.to_string())?;

            state.entity_ids.push(id);
        }

        Operation::UpdateEntity { entity_idx, properties } => {
            if let Some(entity_id) = state.get_entity_id(*entity_idx) {
                let mut tx = db.begin().map_err(|e| e.to_string())?;

                if let Some(mut entity) = tx.get_entity(entity_id).map_err(|e| e.to_string())? {
                    for (key, value) in properties {
                        entity.set_property(key, Value::from(value.clone()));
                    }
                    tx.put_entity(&entity).map_err(|e| e.to_string())?;
                    tx.commit().map_err(|e| e.to_string())?;
                }
            }
        }

        Operation::DeleteEntity { entity_idx } => {
            if let Some(entity_id) = state.get_entity_id(*entity_idx) {
                let mut tx = db.begin().map_err(|e| e.to_string())?;
                tx.delete_entity(entity_id).map_err(|e| e.to_string())?;
                tx.commit().map_err(|e| e.to_string())?;
            }
        }

        Operation::CreateEdge { source_idx, target_idx, edge_type } => {
            let source_id = state.get_entity_id(*source_idx);
            let target_id = state.get_entity_id(*target_idx);

            if let (Some(src), Some(dst)) = (source_id, target_id) {
                if src != dst {
                    let mut tx = db.begin().map_err(|e| e.to_string())?;
                    let edge =
                        tx.create_edge(src, dst, edge_type.as_str()).map_err(|e| e.to_string())?;
                    let edge_id = edge.id;
                    tx.put_edge(&edge).map_err(|e| e.to_string())?;
                    tx.commit().map_err(|e| e.to_string())?;

                    state.edge_info.push((src, dst, edge_id));
                }
            }
        }

        Operation::DeleteEdge { edge_idx } => {
            if let Some((_, _, edge_id)) = state.get_edge_info(*edge_idx) {
                let mut tx = db.begin().map_err(|e| e.to_string())?;
                tx.delete_edge(edge_id).map_err(|e| e.to_string())?;
                tx.commit().map_err(|e| e.to_string())?;
            }
        }

        Operation::ReadEntity { entity_idx } => {
            if let Some(entity_id) = state.get_entity_id(*entity_idx) {
                let tx = db.begin_read().map_err(|e| e.to_string())?;
                let _ = tx.get_entity(entity_id);
            }
        }

        Operation::GetOutgoingEdges { entity_idx } => {
            if let Some(entity_id) = state.get_entity_id(*entity_idx) {
                let tx = db.begin_read().map_err(|e| e.to_string())?;
                let _ = tx.get_outgoing_edges(entity_id);
            }
        }

        Operation::GetIncomingEdges { entity_idx } => {
            if let Some(entity_id) = state.get_entity_id(*entity_idx) {
                let tx = db.begin_read().map_err(|e| e.to_string())?;
                let _ = tx.get_incoming_edges(entity_id);
            }
        }

        Operation::Commit | Operation::Rollback => {
            // These are no-ops in our current execution model
            // (each operation commits immediately)
        }
    }

    Ok(())
}

// ============================================================================
// Property-Based Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Random operations should not panic
    #[test]
    fn test_random_operations_no_panic(operations in operations_strategy(50)) {
        let db = Database::in_memory().expect("failed to create db");
        let _ = execute_operations(&db, &operations);
        // If we get here without panic, the test passes
    }

    /// Database should remain consistent after random operations
    #[test]
    fn test_database_consistency_after_random_ops(operations in operations_strategy(30)) {
        let db = Database::in_memory().expect("failed to create db");
        let state = execute_operations(&db, &operations).expect("execution failed");

        // Verify database is still usable
        let tx = db.begin_read().expect("should be able to read");

        // All remaining entities should be readable
        for &id in &state.entity_ids {
            // May be None if deleted, but shouldn't error
            let _ = tx.get_entity(id).expect("should not error");
        }

        tx.rollback().expect("should be able to rollback");
    }
}

// ============================================================================
// Targeted Fuzz Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Fuzz test for entity creation with random properties
    #[test]
    fn fuzz_entity_creation(
        label in label_strategy(),
        properties in properties_strategy()
    ) {
        let db = Database::in_memory().expect("failed to create db");

        let mut tx = db.begin().expect("failed");
        let mut entity = tx.create_entity().expect("failed");

        if let Some(l) = &label {
            entity = entity.with_label(l.as_str());
        }

        for (key, value) in &properties {
            entity.set_property(key, Value::from(value.clone()));
        }

        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");

        // Verify
        let tx = db.begin_read().expect("failed");
        let retrieved = tx.get_entity(entity.id).expect("failed").expect("should exist");

        if let Some(l) = &label {
            prop_assert!(retrieved.has_label(l.as_str()));
        }

        // Build expected properties (last value for each key wins)
        let mut expected_props = std::collections::HashMap::new();
        for (key, value) in &properties {
            expected_props.insert(key.clone(), Value::from(value.clone()));
        }

        for (key, expected) in &expected_props {
            prop_assert_eq!(retrieved.get_property(key), Some(expected));
        }
    }

    /// Fuzz test for edge creation
    #[test]
    fn fuzz_edge_creation(
        edge_type in edge_type_strategy(),
        properties in properties_strategy()
    ) {
        let db = Database::in_memory().expect("failed to create db");

        // Create entities
        let mut tx = db.begin().expect("failed");
        let src = tx.create_entity().expect("failed");
        let dst = tx.create_entity().expect("failed");
        tx.put_entity(&src).expect("failed");
        tx.put_entity(&dst).expect("failed");

        let mut edge = tx.create_edge(src.id, dst.id, edge_type.as_str()).expect("failed");
        for (key, value) in &properties {
            edge.set_property(key, Value::from(value.clone()));
        }
        tx.put_edge(&edge).expect("failed");
        tx.commit().expect("failed");

        // Verify
        let tx = db.begin_read().expect("failed");
        let edges = tx.get_outgoing_edges(src.id).expect("failed");

        prop_assert_eq!(edges.len(), 1);
        prop_assert_eq!(edges[0].edge_type.as_str(), edge_type);

        // Build expected properties (last value for each key wins)
        let mut expected_props = std::collections::HashMap::new();
        for (key, value) in &properties {
            expected_props.insert(key.clone(), Value::from(value.clone()));
        }

        for (key, expected) in &expected_props {
            prop_assert_eq!(edges[0].get_property(key), Some(expected));
        }
    }

    /// Fuzz test for vector properties
    #[test]
    fn fuzz_vector_properties(
        vector in proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1..128
        )
    ) {
        let db = Database::in_memory().expect("failed to create db");

        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity()
            .expect("failed")
            .with_property("embedding", vector.clone());
        let id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");

        // Verify
        let tx = db.begin_read().expect("failed");
        let retrieved = tx.get_entity(id).expect("failed").expect("should exist");

        if let Some(Value::Vector(v)) = retrieved.get_property("embedding") {
            prop_assert_eq!(v.len(), vector.len());
            for (a, b) in v.iter().zip(vector.iter()) {
                prop_assert!((a - b).abs() < f32::EPSILON);
            }
        } else {
            prop_assert!(false, "expected vector property");
        }
    }
}
