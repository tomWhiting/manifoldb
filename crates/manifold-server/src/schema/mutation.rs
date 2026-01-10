//! GraphQL mutation resolvers.
//!
//! These resolvers handle write operations against the database.

use std::sync::Arc;

use async_graphql::{Context, Object, Result, ID};
use manifoldb::{Database, DistanceMetric, Entity, EntityId};

use super::types::{
    CollectionInfo, CreateCollectionInput, CreateEdgeInput, CreateNodeInput, DistanceMetricEnum,
    Edge, Node, QueryResult, UpsertVectorInput, VectorConfigInfo, VectorTypeEnum,
};
use crate::convert;

/// Root mutation type for the GraphQL schema.
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Execute a write query (INSERT, UPDATE, DELETE, CREATE, etc.).
    async fn execute(&self, ctx: &Context<'_>, query: String) -> Result<QueryResult> {
        let db = ctx.data::<Arc<Database>>()?;
        let result = db.query(&query)?;

        Ok(QueryResult {
            table: Some(convert::query_result_to_table(&result)),
            graph: None,
        })
    }

    /// Create a node with labels and properties.
    async fn create_node(&self, ctx: &Context<'_>, input: CreateNodeInput) -> Result<Node> {
        let db = ctx.data::<Arc<Database>>()?;

        let labels_str = input.labels.join(":");
        let props = input
            .properties
            .map(|p| format!(" {}", p.0))
            .unwrap_or_default();

        let query = format!("CREATE (n:{}{}) RETURN n", labels_str, props);
        let result = db.query(&query)?;

        let graph = convert::query_result_to_graph(&result)?;
        graph
            .nodes
            .into_iter()
            .next()
            .ok_or_else(|| async_graphql::Error::new("Failed to create node"))
    }

    /// Create an edge between two nodes.
    async fn create_edge(&self, ctx: &Context<'_>, input: CreateEdgeInput) -> Result<Edge> {
        let db = ctx.data::<Arc<Database>>()?;

        let props = input
            .properties
            .map(|p| format!(" {}", p.0))
            .unwrap_or_default();

        let query = format!(
            "MATCH (a), (b) WHERE id(a) = {} AND id(b) = {} \
             CREATE (a)-[r:{}{}]->(b) RETURN r",
            input.source_id, input.target_id, input.edge_type, props
        );

        let result = db.query(&query)?;

        let graph = convert::query_result_to_graph(&result)?;
        graph
            .edges
            .into_iter()
            .next()
            .ok_or_else(|| async_graphql::Error::new("Failed to create edge"))
    }

    /// Delete a node by ID (and its connected edges).
    async fn delete_node(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let db = ctx.data::<Arc<Database>>()?;
        let query = format!("MATCH (n) WHERE id(n) = {} DETACH DELETE n", id.as_str());
        db.execute(&query)?;
        Ok(true)
    }

    /// Delete an edge by ID.
    async fn delete_edge(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let db = ctx.data::<Arc<Database>>()?;
        let query = format!("MATCH ()-[r]->() WHERE id(r) = {} DELETE r", id.as_str());
        db.execute(&query)?;
        Ok(true)
    }

    /// Update node properties (merge with existing).
    async fn update_node(
        &self,
        ctx: &Context<'_>,
        id: ID,
        properties: async_graphql::Json<serde_json::Value>,
    ) -> Result<Node> {
        let db = ctx.data::<Arc<Database>>()?;

        let query = format!(
            "MATCH (n) WHERE id(n) = {} SET n += {} RETURN n",
            id.as_str(), properties.0
        );

        let result = db.query(&query)?;

        let graph = convert::query_result_to_graph(&result)?;
        graph
            .nodes
            .into_iter()
            .next()
            .ok_or_else(|| async_graphql::Error::new("Node not found"))
    }

    // =========================================================================
    // Vector/Collection Mutations
    // =========================================================================

    /// Create a new vector collection.
    async fn create_collection(
        &self,
        ctx: &Context<'_>,
        input: CreateCollectionInput,
    ) -> Result<CollectionInfo> {
        let db = ctx.data::<Arc<Database>>()?;

        // Start building the collection
        let mut builder = db.create_collection(&input.name)?;

        // Add vector configurations
        for vec_config in &input.vectors {
            let distance = vec_config
                .distance_metric
                .map(graphql_distance_to_manifold)
                .unwrap_or(DistanceMetric::Cosine);

            builder = builder.with_dense_vector(&vec_config.name, vec_config.dimension as usize, distance);
        }

        // Build the collection
        let handle = builder.build()?;

        // Return collection info
        let vectors = handle.vectors();
        let point_count = handle.count_points().unwrap_or(0);

        let vector_configs: Vec<VectorConfigInfo> = vectors
            .iter()
            .map(|(vec_name, config)| {
                let (vector_type, dimension) = match &config.vector_type {
                    manifoldb::collection::VectorType::Dense { dimension } => {
                        (VectorTypeEnum::Dense, Some(*dimension as i32))
                    }
                    manifoldb::collection::VectorType::Sparse { .. } => (VectorTypeEnum::Sparse, None),
                    manifoldb::collection::VectorType::Multi { token_dim } => {
                        (VectorTypeEnum::Multi, Some(*token_dim as i32))
                    }
                    manifoldb::collection::VectorType::Binary { bits } => {
                        (VectorTypeEnum::Binary, Some(*bits as i32))
                    }
                };

                let distance_metric = match &config.distance {
                    manifoldb::collection::DistanceType::Dense(m) => match m {
                        DistanceMetric::Cosine => DistanceMetricEnum::Cosine,
                        DistanceMetric::DotProduct => DistanceMetricEnum::DotProduct,
                        DistanceMetric::Euclidean => DistanceMetricEnum::Euclidean,
                        DistanceMetric::Manhattan => DistanceMetricEnum::Manhattan,
                        DistanceMetric::Chebyshev => DistanceMetricEnum::Chebyshev,
                    },
                    manifoldb::collection::DistanceType::Sparse(_) => DistanceMetricEnum::DotProduct,
                    manifoldb::collection::DistanceType::Binary(_) => DistanceMetricEnum::Hamming,
                };

                VectorConfigInfo {
                    name: vec_name.clone(),
                    vector_type,
                    dimension,
                    distance_metric,
                }
            })
            .collect();

        Ok(CollectionInfo {
            name: input.name,
            vectors: vector_configs,
            point_count: point_count as i32,
        })
    }

    /// Delete a vector collection.
    async fn delete_collection(&self, ctx: &Context<'_>, name: String) -> Result<bool> {
        let db = ctx.data::<Arc<Database>>()?;
        db.drop_collection(&name)?;
        Ok(true)
    }

    /// Upsert a vector into a collection.
    async fn upsert_vector(
        &self,
        ctx: &Context<'_>,
        collection: String,
        input: UpsertVectorInput,
    ) -> Result<ID> {
        let db = ctx.data::<Arc<Database>>()?;

        // Generate or parse the ID
        let entity_id = if let Some(ref id) = input.id {
            let id_str = id.as_str();
            id_str
                .parse::<u64>()
                .map(EntityId::new)
                .map_err(|_| async_graphql::Error::new(format!("Invalid ID: {}", id_str)))?
        } else {
            // Generate a new ID using current timestamp + random component
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            EntityId::new(ts)
        };

        // Build the entity
        let mut entity = Entity::new(entity_id);

        // Add payload as properties
        if let Some(ref payload) = input.payload {
            if let Some(obj) = payload.0.as_object() {
                for (key, value) in obj {
                    if let Some(prop_value) = json_to_value(value) {
                        entity = entity.with_property(key.clone(), prop_value);
                    }
                }
            }
        }

        // Add vectors
        for vec_input in &input.vectors {
            let vector: Vec<f32> = vec_input.values.iter().map(|v| *v as f32).collect();
            entity = entity.with_vector(vec_input.name.clone(), vector);
        }

        // Upsert the entity
        db.upsert(&collection, &entity)?;

        Ok(ID(entity_id.as_u64().to_string()))
    }
}

/// Convert GraphQL distance metric to ManifoldDB distance metric.
fn graphql_distance_to_manifold(metric: DistanceMetricEnum) -> DistanceMetric {
    match metric {
        DistanceMetricEnum::Cosine => DistanceMetric::Cosine,
        DistanceMetricEnum::DotProduct => DistanceMetric::DotProduct,
        DistanceMetricEnum::Euclidean => DistanceMetric::Euclidean,
        DistanceMetricEnum::Manhattan => DistanceMetric::Manhattan,
        DistanceMetricEnum::Chebyshev => DistanceMetric::Chebyshev,
        DistanceMetricEnum::Hamming => DistanceMetric::Cosine, // Binary uses Hamming but we default to Cosine for dense
    }
}

/// Convert JSON value to ManifoldDB Value.
fn json_to_value(json: &serde_json::Value) -> Option<manifoldb::Value> {
    match json {
        serde_json::Value::Null => Some(manifoldb::Value::Null),
        serde_json::Value::Bool(b) => Some(manifoldb::Value::Bool(*b)),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(manifoldb::Value::Int)
            .or_else(|| n.as_f64().map(manifoldb::Value::Float)),
        serde_json::Value::String(s) => Some(manifoldb::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            // Check if it's a vector (all f32)
            let floats: Option<Vec<f32>> = arr
                .iter()
                .map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            if let Some(vec) = floats {
                Some(manifoldb::Value::Vector(vec))
            } else {
                // Try as array of values
                let values: Option<Vec<manifoldb::Value>> =
                    arr.iter().map(json_to_value).collect();
                values.map(manifoldb::Value::Array)
            }
        }
        serde_json::Value::Object(_) => {
            // Nested objects not directly supported
            None
        }
    }
}
