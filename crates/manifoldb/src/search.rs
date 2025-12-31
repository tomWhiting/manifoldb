//! Unified vector search API.
//!
//! This module provides the unified search API that works with entities
//! instead of collection-specific point types.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::{Database, Filter, ScoredEntity};
//!
//! let db = Database::in_memory()?;
//!
//! // Create a collection with a dense vector
//! db.create_collection("documents")?
//!     .with_dense_vector("embedding", 768, DistanceMetric::Cosine)
//!     .build()?;
//!
//! // Upsert entities with vectors
//! let entity = Entity::new(EntityId::new(1))
//!     .with_label("Document")
//!     .with_property("title", "Hello World")
//!     .with_vector("embedding", embedding);
//! db.upsert("documents", entity)?;
//!
//! // Search returns ScoredEntity
//! let results: Vec<ScoredEntity> = db.search("documents", "embedding")
//!     .query(query_vector)
//!     .filter(Filter::eq("title", "Hello World"))
//!     .limit(10)
//!     .execute()?;
//! ```
//!
//! # Graph-Constrained Search
//!
//! You can constrain vector search results to entities reachable via a graph
//! traversal pattern using [`within_traversal()`](EntitySearchBuilder::within_traversal):
//!
//! ```ignore
//! // Search for similar symbols, but only within a specific repository
//! let results = db.search("symbols", "embedding")
//!     .query(query_vector)
//!     .within_traversal(repo_id, |p| p
//!         .edge_out("CONTAINS")
//!         .variable_length(1, 10)
//!     )
//!     .filter(Filter::eq("visibility", "public"))
//!     .limit(10)
//!     .execute()?;
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use manifoldb_core::{Entity, EntityId, ScoredEntity, VectorData};
use manifoldb_graph::traversal::{Direction, PathPattern, PathStep};
use manifoldb_storage::backends::RedbEngine;

use crate::collection::CollectionHandle;
use crate::collection::Vector as CollectionVector;
use crate::error::Result;
use crate::filter::Filter;
use crate::Error;

/// A constraint that limits vector search results to entities reachable
/// via a graph traversal.
///
/// This is used internally by [`EntitySearchBuilder::within_traversal()`].
#[derive(Clone)]
pub struct TraversalConstraint {
    /// The starting entity ID for the traversal.
    start: EntityId,
    /// The path pattern to match during traversal.
    pattern: PathPattern,
}

impl TraversalConstraint {
    /// Create a new traversal constraint.
    pub fn new(start: EntityId, pattern: PathPattern) -> Self {
        Self { start, pattern }
    }

    /// Get the starting entity ID.
    #[must_use]
    pub fn start(&self) -> EntityId {
        self.start
    }

    /// Get the path pattern.
    #[must_use]
    pub fn pattern(&self) -> &PathPattern {
        &self.pattern
    }
}

/// Builder for constructing graph traversal patterns.
///
/// This provides a fluent API for defining path patterns that constrain
/// which entities are considered in a vector search.
///
/// # Example
///
/// ```ignore
/// // Pattern: (start)-[:CONTAINS*1..10]->(result)
/// let builder = TraversalPatternBuilder::new()
///     .edge_out("CONTAINS")
///     .variable_length(1, 10);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TraversalPatternBuilder {
    pattern: PathPattern,
}

impl TraversalPatternBuilder {
    /// Create a new empty pattern builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an outgoing edge step with the specified type.
    ///
    /// This matches edges where the pattern traverses from source to target.
    #[must_use]
    pub fn edge_out(mut self, edge_type: impl Into<manifoldb_core::EdgeType>) -> Self {
        self.pattern = self.pattern.add_step(PathStep::outgoing(edge_type));
        self
    }

    /// Add an incoming edge step with the specified type.
    ///
    /// This matches edges where the pattern traverses from target to source.
    #[must_use]
    pub fn edge_in(mut self, edge_type: impl Into<manifoldb_core::EdgeType>) -> Self {
        self.pattern = self.pattern.add_step(PathStep::incoming(edge_type));
        self
    }

    /// Add a bidirectional edge step with the specified type.
    ///
    /// This matches edges in either direction.
    #[must_use]
    pub fn edge_both(mut self, edge_type: impl Into<manifoldb_core::EdgeType>) -> Self {
        self.pattern = self.pattern.add_step(PathStep::both(edge_type));
        self
    }

    /// Add an outgoing edge step that matches any edge type.
    #[must_use]
    pub fn any_out(mut self) -> Self {
        self.pattern = self.pattern.add_step(PathStep::any(Direction::Outgoing));
        self
    }

    /// Add an incoming edge step that matches any edge type.
    #[must_use]
    pub fn any_in(mut self) -> Self {
        self.pattern = self.pattern.add_step(PathStep::any(Direction::Incoming));
        self
    }

    /// Add a bidirectional edge step that matches any edge type.
    #[must_use]
    pub fn any_both(mut self) -> Self {
        self.pattern = self.pattern.add_step(PathStep::any(Direction::Both));
        self
    }

    /// Make the last step variable-length with the given hop range.
    ///
    /// This is equivalent to Cypher's `*min..max` syntax.
    ///
    /// # Panics
    ///
    /// Panics if no steps have been added yet.
    #[must_use]
    pub fn variable_length(mut self, min: usize, max: usize) -> Self {
        let steps = self.pattern.steps();
        if steps.is_empty() {
            return self;
        }

        // Replace the last step with a variable-length version
        let last_idx = steps.len() - 1;
        let last_step = steps[last_idx].clone();
        let var_step = PathStep::new(last_step.direction, last_step.filter.clone())
            .variable_length(min, max);

        // Rebuild pattern with modified last step
        let mut new_pattern = PathPattern::new();
        for (i, step) in steps.iter().enumerate() {
            if i == last_idx {
                new_pattern = new_pattern.add_step(var_step.clone());
            } else {
                new_pattern = new_pattern.add_step(step.clone());
            }
        }
        self.pattern = new_pattern;
        self
    }

    /// Add a custom step with full control over direction and filter.
    #[must_use]
    pub fn step(mut self, step: PathStep) -> Self {
        self.pattern = self.pattern.add_step(step);
        self
    }

    /// Build the final path pattern.
    #[must_use]
    pub fn build(self) -> PathPattern {
        self.pattern
    }
}

/// Builder for unified vector search operations.
///
/// The `EntitySearchBuilder` provides a fluent interface for constructing
/// vector similarity searches that return entities instead of points.
///
/// # Example
///
/// ```ignore
/// use manifoldb::{Database, Filter};
///
/// let results = db.search("documents", "embedding")
///     .query(vec![0.1, 0.2, 0.3])
///     .filter(Filter::eq("language", "rust"))
///     .limit(10)
///     .execute()?;
///
/// for result in results {
///     println!("{}: {:.4}", result.entity.id.as_u64(), result.score);
/// }
/// ```
pub struct EntitySearchBuilder {
    /// The collection handle (owned).
    handle: CollectionHandle<Arc<RedbEngine>>,
    /// The storage engine for graph traversal.
    engine: Arc<RedbEngine>,
    /// The vector name to search.
    vector_name: String,
    /// The query vector.
    query: Option<VectorData>,
    /// Maximum number of results.
    limit: usize,
    /// Number of results to skip.
    offset: usize,
    /// Optional filter expression.
    filter: Option<Filter>,
    /// Minimum score threshold.
    score_threshold: Option<f32>,
    /// Optional traversal constraint for graph-bounded search.
    traversal_constraint: Option<TraversalConstraint>,
}

impl EntitySearchBuilder {
    /// Create a new search builder.
    pub(crate) fn new(
        handle: CollectionHandle<Arc<RedbEngine>>,
        engine: Arc<RedbEngine>,
        vector_name: impl Into<String>,
    ) -> Self {
        Self {
            handle,
            engine,
            vector_name: vector_name.into(),
            query: None,
            limit: 10,
            offset: 0,
            filter: None,
            score_threshold: None,
            traversal_constraint: None,
        }
    }

    /// Set the query vector.
    ///
    /// The vector can be a dense vector (`Vec<f32>`), sparse vector, or
    /// any type that converts to `VectorData`.
    #[must_use]
    pub fn query(mut self, vector: impl Into<VectorData>) -> Self {
        self.query = Some(vector.into());
        self
    }

    /// Set the maximum number of results to return.
    ///
    /// Default is 10.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the number of results to skip.
    ///
    /// Useful for pagination.
    #[must_use]
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Add a filter to narrow results.
    ///
    /// The filter is applied to entity properties.
    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a minimum score threshold.
    ///
    /// Results with scores below this threshold are excluded.
    #[must_use]
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    /// Constrain search results to entities reachable via a graph traversal.
    ///
    /// This enables graph-bounded vector search, where only entities that
    /// are reachable from the starting entity via the specified path pattern
    /// are considered as search results.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting entity ID for the traversal
    /// * `pattern_builder` - A closure that builds the traversal pattern
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, EntityId};
    ///
    /// // Search for similar symbols within a specific repository
    /// let repo_id = EntityId::new(123);
    /// let results = db.search("symbols", "embedding")
    ///     .query(query_vector)
    ///     .within_traversal(repo_id, |p| p
    ///         .edge_out("CONTAINS")
    ///         .variable_length(1, 10)
    ///     )
    ///     .limit(10)
    ///     .execute()?;
    /// ```
    ///
    /// # How It Works
    ///
    /// 1. The traversal is executed first to find all reachable entity IDs
    /// 2. Vector search is performed normally
    /// 3. Results are filtered to only include entities in the traversal set
    ///
    /// This approach ensures that vector search results are constrained to
    /// the graph structure while maintaining vector similarity ordering.
    #[must_use]
    pub fn within_traversal<F>(mut self, start: EntityId, pattern_builder: F) -> Self
    where
        F: FnOnce(TraversalPatternBuilder) -> TraversalPatternBuilder,
    {
        let builder = TraversalPatternBuilder::new();
        let pattern = pattern_builder(builder).build();
        self.traversal_constraint = Some(TraversalConstraint::new(start, pattern));
        self
    }

    /// Constrain search results using a pre-built traversal constraint.
    ///
    /// This is an alternative to [`within_traversal()`](Self::within_traversal)
    /// that accepts a pre-built [`TraversalConstraint`].
    #[must_use]
    pub fn with_traversal_constraint(mut self, constraint: TraversalConstraint) -> Self {
        self.traversal_constraint = Some(constraint);
        self
    }

    /// Execute the search and return scored entities.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No query vector was provided
    /// - The vector name doesn't exist in the collection
    /// - A storage error occurs
    /// - A graph traversal error occurs (when using `within_traversal`)
    pub fn execute(self) -> Result<Vec<ScoredEntity>> {
        use manifoldb_storage::StorageEngine;

        let query =
            self.query.ok_or_else(|| Error::InvalidInput("No query vector provided".into()))?;

        // If we have a traversal constraint, execute the traversal first
        // to get the set of reachable entity IDs
        let reachable_ids: Option<HashSet<EntityId>> =
            if let Some(ref constraint) = self.traversal_constraint {
                let tx = self.engine.begin_read().map_err(|e| {
                    Error::Execution(format!("Failed to start read transaction: {e}"))
                })?;

                let matches = constraint
                    .pattern()
                    .find_from(&tx, constraint.start())
                    .map_err(|e| Error::Execution(format!("Graph traversal failed: {e}")))?;

                // Collect all target entity IDs from the traversal matches
                let ids: HashSet<EntityId> = matches.iter().map(|m| m.target()).collect();

                Some(ids)
            } else {
                None
            };

        // Convert VectorData to collection Vector
        let collection_query = vector_data_to_collection_vector(&query);

        // Convert our Filter to collection Filter
        let collection_filter = self.filter.map(filter_to_collection_filter);

        // When we have a traversal constraint, we need to fetch more results
        // than the limit, since we'll filter them down afterward.
        // We use a heuristic: fetch up to 10x the limit or at least 100 results.
        let fetch_limit = if self.traversal_constraint.is_some() {
            (self.limit * 10).max(100)
        } else {
            self.limit
        };

        // Execute search using collection handle
        let scored_points = self
            .handle
            .execute_search(
                &self.vector_name,
                collection_query,
                fetch_limit,
                self.offset,
                collection_filter,
                true,  // with_payload - we need properties
                false, // with_vectors - vectors are in Entity.vectors
                self.score_threshold,
                None, // ef
            )
            .map_err(|e| Error::Collection(e.to_string()))?;

        // Convert ScoredPoint to ScoredEntity and filter by traversal if applicable
        let results: Vec<ScoredEntity> = if let Some(ref allowed_ids) = reachable_ids {
            scored_points
                .into_iter()
                .filter_map(|sp| {
                    let entity_id = EntityId::new(sp.id.as_u64());
                    if allowed_ids.contains(&entity_id) {
                        let entity = scored_point_to_entity(sp.id, sp.payload);
                        Some(ScoredEntity::new(entity, sp.score))
                    } else {
                        None
                    }
                })
                .take(self.limit)
                .collect()
        } else {
            scored_points
                .into_iter()
                .map(|sp| {
                    let entity = scored_point_to_entity(sp.id, sp.payload);
                    ScoredEntity::new(entity, sp.score)
                })
                .collect()
        };

        Ok(results)
    }
}

/// Convert VectorData to collection Vector.
fn vector_data_to_collection_vector(data: &VectorData) -> CollectionVector {
    match data {
        VectorData::Dense(v) => CollectionVector::Dense(v.clone()),
        VectorData::Sparse(v) => CollectionVector::Sparse(v.clone()),
        VectorData::Multi(v) => CollectionVector::Multi(v.clone()),
    }
}

/// Convert our Filter to collection Filter.
fn filter_to_collection_filter(filter: Filter) -> crate::collection::Filter {
    match filter {
        Filter::Eq { field, value } => crate::collection::Filter::Eq { field, value },
        Filter::Ne { field, value } => crate::collection::Filter::Ne { field, value },
        Filter::Gt { field, value } => crate::collection::Filter::Gt { field, value },
        Filter::Gte { field, value } => crate::collection::Filter::Gte { field, value },
        Filter::Lt { field, value } => crate::collection::Filter::Lt { field, value },
        Filter::Lte { field, value } => crate::collection::Filter::Lte { field, value },
        Filter::Range { field, min, max } => crate::collection::Filter::Range { field, min, max },
        Filter::In { field, values } => crate::collection::Filter::In { field, values },
        Filter::NotIn { field, values } => crate::collection::Filter::NotIn { field, values },
        Filter::Contains { field, substring } => {
            crate::collection::Filter::Contains { field, substring }
        }
        Filter::StartsWith { field, prefix } => {
            crate::collection::Filter::StartsWith { field, prefix }
        }
        Filter::ArrayContains { field, value } => {
            crate::collection::Filter::ArrayContains { field, value }
        }
        Filter::Exists { field } => crate::collection::Filter::Exists { field },
        Filter::NotExists { field } => crate::collection::Filter::NotExists { field },
        Filter::And(filters) => crate::collection::Filter::And(
            filters.into_iter().map(filter_to_collection_filter).collect(),
        ),
        Filter::Or(filters) => crate::collection::Filter::Or(
            filters.into_iter().map(filter_to_collection_filter).collect(),
        ),
        Filter::Not(filter) => {
            crate::collection::Filter::Not(Box::new(filter_to_collection_filter(*filter)))
        }
    }
}

/// Convert a ScoredPoint's data to an Entity.
fn scored_point_to_entity(
    id: manifoldb_core::PointId,
    payload: Option<serde_json::Value>,
) -> Entity {
    let entity_id = EntityId::new(id.as_u64());
    let mut entity = Entity::new(entity_id);

    // Convert payload to entity properties
    if let Some(serde_json::Value::Object(map)) = payload {
        for (key, value) in map {
            if let Some(prop_value) = json_to_value(&value) {
                entity = entity.with_property(key, prop_value);
            }
        }
    }

    entity
}

/// Convert JSON value to manifoldb Value.
fn json_to_value(json: &serde_json::Value) -> Option<manifoldb_core::Value> {
    match json {
        serde_json::Value::Null => Some(manifoldb_core::Value::Null),
        serde_json::Value::Bool(b) => Some(manifoldb_core::Value::Bool(*b)),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(manifoldb_core::Value::Int)
            .or_else(|| n.as_f64().map(manifoldb_core::Value::Float)),
        serde_json::Value::String(s) => Some(manifoldb_core::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            // Check if it's a vector (all f32)
            let floats: Option<Vec<f32>> =
                arr.iter().map(|v| v.as_f64().map(|f| f as f32)).collect();
            if let Some(vec) = floats {
                Some(manifoldb_core::Value::Vector(vec))
            } else {
                // Try as array of values
                let values: Option<Vec<manifoldb_core::Value>> =
                    arr.iter().map(json_to_value).collect();
                values.map(manifoldb_core::Value::Array)
            }
        }
        serde_json::Value::Object(_) => {
            // Nested objects not directly supported
            None
        }
    }
}

/// Convert Entity to collection PointStruct for upserting.
pub fn entity_to_point_struct(
    entity: &Entity,
    collection_name: &str,
) -> crate::collection::PointStruct {
    use crate::collection::PointStruct;

    let mut point = PointStruct::new(entity.id.as_u64());

    // Convert properties to payload
    let payload = entity_properties_to_json(entity);
    if !payload.as_object().map_or(true, |o| o.is_empty()) {
        point = point.with_payload(payload);
    }

    // Convert vectors
    for (name, vector_data) in &entity.vectors {
        let collection_vec = match vector_data {
            VectorData::Dense(v) => CollectionVector::Dense(v.clone()),
            VectorData::Sparse(v) => CollectionVector::Sparse(v.clone()),
            VectorData::Multi(v) => CollectionVector::Multi(v.clone()),
        };
        point = point.with_vector(name.clone(), collection_vec);
    }

    // Store collection name as metadata (for potential use in queries)
    let _ = collection_name; // Currently unused but may be needed for multi-collection support

    point
}

/// Convert entity properties to JSON.
fn entity_properties_to_json(entity: &Entity) -> serde_json::Value {
    let mut map = serde_json::Map::new();

    // Add labels as a property for filtering
    if !entity.labels.is_empty() {
        let labels: Vec<serde_json::Value> = entity
            .labels
            .iter()
            .map(|l| serde_json::Value::String(l.as_str().to_string()))
            .collect();
        map.insert("_labels".to_string(), serde_json::Value::Array(labels));
    }

    // Add properties
    for (key, value) in &entity.properties {
        map.insert(key.clone(), value_to_json(value));
    }

    serde_json::Value::Object(map)
}

/// Convert manifoldb Value to JSON value.
fn value_to_json(value: &manifoldb_core::Value) -> serde_json::Value {
    match value {
        manifoldb_core::Value::Null => serde_json::Value::Null,
        manifoldb_core::Value::Bool(b) => serde_json::Value::Bool(*b),
        manifoldb_core::Value::Int(i) => serde_json::json!(*i),
        manifoldb_core::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        manifoldb_core::Value::String(s) => serde_json::Value::String(s.clone()),
        manifoldb_core::Value::Bytes(b) => {
            use base64::Engine;
            serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        manifoldb_core::Value::Vector(v) => {
            serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
        }
        manifoldb_core::Value::SparseVector(pairs) => serde_json::Value::Array(
            pairs.iter().map(|(idx, val)| serde_json::json!([*idx, *val])).collect(),
        ),
        manifoldb_core::Value::MultiVector(vecs) => serde_json::Value::Array(
            vecs.iter()
                .map(|v| {
                    serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
                })
                .collect(),
        ),
        manifoldb_core::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(value_to_json).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_to_value_primitives() {
        assert_eq!(json_to_value(&serde_json::json!(null)), Some(manifoldb_core::Value::Null));
        assert_eq!(
            json_to_value(&serde_json::json!(true)),
            Some(manifoldb_core::Value::Bool(true))
        );
        assert_eq!(json_to_value(&serde_json::json!(42)), Some(manifoldb_core::Value::Int(42)));
        assert_eq!(
            json_to_value(&serde_json::json!(3.14)),
            Some(manifoldb_core::Value::Float(3.14))
        );
        assert_eq!(
            json_to_value(&serde_json::json!("hello")),
            Some(manifoldb_core::Value::String("hello".to_string()))
        );
    }

    #[test]
    fn test_value_to_json_roundtrip() {
        let original = manifoldb_core::Value::String("test".to_string());
        let json = value_to_json(&original);
        let recovered = json_to_value(&json);
        assert_eq!(recovered, Some(original));
    }

    #[test]
    fn test_entity_properties_to_json() {
        let entity = Entity::new(EntityId::new(1))
            .with_label("Test")
            .with_property("name", "Alice")
            .with_property("age", 30i64);

        let json = entity_properties_to_json(&entity);
        let obj = json.as_object().expect("should be object");

        assert!(obj.contains_key("_labels"));
        assert!(obj.contains_key("name"));
        assert!(obj.contains_key("age"));
    }
}
