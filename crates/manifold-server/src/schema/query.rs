//! GraphQL query resolvers.
//!
//! These resolvers handle read operations against the database.

use std::sync::Arc;

use async_graphql::{Context, Object, Result, ID};
use manifoldb::Database;

use super::types::{
    CollectionInfo, Direction, DistanceMetricEnum, EdgeTypeInfo, GraphResult, GraphStats,
    LabelInfo, Node, TableResult, VectorConfigInfo, VectorSearchInput, VectorSearchResult,
    VectorTypeEnum,
};
use crate::convert;

/// Root query type for the GraphQL schema.
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Execute a SQL query and return tabular results.
    async fn sql(&self, ctx: &Context<'_>, query: String) -> Result<TableResult> {
        let db = ctx.data::<Arc<Database>>()?;
        let result = db.query(&query)?;
        Ok(convert::query_result_to_table(&result))
    }

    /// Execute a Cypher query and return graph results.
    async fn cypher(&self, ctx: &Context<'_>, query: String) -> Result<GraphResult> {
        let db = ctx.data::<Arc<Database>>()?;
        let result = db.query(&query)?;
        convert::query_result_to_graph(&result)
    }

    /// Get database graph statistics.
    async fn stats(&self, ctx: &Context<'_>) -> Result<GraphStats> {
        let db = ctx.data::<Arc<Database>>()?;
        let stats = db.graph_stats()?;
        Ok(GraphStats {
            node_count: stats.node_count as i64,
            edge_count: stats.edge_count as i64,
            labels: stats
                .labels
                .into_iter()
                .map(|(name, count)| LabelInfo { name, count: count as i64 })
                .collect(),
            edge_types: stats
                .edge_types
                .into_iter()
                .map(|(name, count)| EdgeTypeInfo { name, count: count as i64 })
                .collect(),
        })
    }

    /// Get all node labels with counts.
    async fn labels(&self, ctx: &Context<'_>) -> Result<Vec<LabelInfo>> {
        let db = ctx.data::<Arc<Database>>()?;
        let labels = db.list_labels()?;
        Ok(labels
            .into_iter()
            .map(|(name, count)| LabelInfo { name, count: count as i64 })
            .collect())
    }

    /// Get all edge types with counts.
    async fn edge_types(&self, ctx: &Context<'_>) -> Result<Vec<EdgeTypeInfo>> {
        let db = ctx.data::<Arc<Database>>()?;
        let edge_types = db.list_edge_types()?;
        Ok(edge_types
            .into_iter()
            .map(|(name, count)| EdgeTypeInfo { name, count: count as i64 })
            .collect())
    }

    /// Get a single node by ID.
    async fn node(&self, ctx: &Context<'_>, id: ID) -> Result<Option<Node>> {
        let db = ctx.data::<Arc<Database>>()?;
        let query = format!("MATCH (n) WHERE id(n) = {} RETURN n", id.as_str());
        let result = db.query(&query)?;

        if result.is_empty() {
            return Ok(None);
        }

        let graph = convert::query_result_to_graph(&result)?;
        Ok(graph.nodes.into_iter().next())
    }

    /// Get nodes by label with optional limit.
    async fn nodes_by_label(
        &self,
        ctx: &Context<'_>,
        label: String,
        limit: Option<i32>,
    ) -> Result<Vec<Node>> {
        let db = ctx.data::<Arc<Database>>()?;
        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
        let query = format!("MATCH (n:{}) RETURN n{}", label, limit_clause);
        let result = db.query(&query)?;
        let graph = convert::query_result_to_graph(&result)?;
        Ok(graph.nodes)
    }

    /// Get neighbors of a node.
    async fn neighbors(
        &self,
        ctx: &Context<'_>,
        node_id: ID,
        direction: Option<Direction>,
        edge_type: Option<String>,
        limit: Option<i32>,
    ) -> Result<GraphResult> {
        let db = ctx.data::<Arc<Database>>()?;

        let type_filter = edge_type.map(|t| format!(":{}", t)).unwrap_or_default();

        let (left_arrow, right_arrow) = match direction.unwrap_or(Direction::Both) {
            Direction::Outgoing => ("", "->"),
            Direction::Incoming => ("<-", ""),
            Direction::Both => ("", "-"),
        };

        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();

        let query = format!(
            "MATCH (n){}[r{}]{}(m) WHERE id(n) = {} RETURN n, r, m{}",
            left_arrow,
            type_filter,
            right_arrow,
            node_id.as_str(),
            limit_clause
        );

        let result = db.query(&query)?;
        convert::query_result_to_graph(&result)
    }

    // =========================================================================
    // Vector/Collection Queries
    // =========================================================================

    /// List all vector collections.
    async fn collections(&self, ctx: &Context<'_>) -> Result<Vec<CollectionInfo>> {
        let db = ctx.data::<Arc<Database>>()?;
        let collection_names = db.list_collections()?;

        let mut collections = Vec::new();
        for name in collection_names {
            if let Ok(info) = get_collection_info(db, &name) {
                collections.push(info);
            }
        }

        Ok(collections)
    }

    /// Get a single vector collection by name.
    async fn collection(&self, ctx: &Context<'_>, name: String) -> Result<Option<CollectionInfo>> {
        let db = ctx.data::<Arc<Database>>()?;

        match get_collection_info(db, &name) {
            Ok(info) => Ok(Some(info)),
            Err(_) => Ok(None),
        }
    }

    /// Search for similar vectors in a collection.
    async fn search_similar(
        &self,
        ctx: &Context<'_>,
        collection: String,
        input: VectorSearchInput,
    ) -> Result<Vec<VectorSearchResult>> {
        let db = ctx.data::<Arc<Database>>()?;

        // Convert f64 query vector to f32
        let query_vector: Vec<f32> = input.query_vector.iter().map(|v| *v as f32).collect();

        // Build the search
        let mut search = db.search(&collection, &input.vector_name)?;
        search = search.query(query_vector);

        if let Some(limit) = input.limit {
            search = search.limit(limit as usize);
        }

        if let Some(offset) = input.offset {
            search = search.offset(offset as usize);
        }

        if let Some(threshold) = input.score_threshold {
            search = search.score_threshold(threshold as f32);
        }

        // Execute the search
        let results = search.execute()?;

        // Determine whether to include payload in results
        let include_payload = input.with_payload.unwrap_or(true);

        // Convert to GraphQL results
        let search_results: Vec<VectorSearchResult> = results
            .into_iter()
            .map(|scored| {
                let payload = if include_payload && !scored.entity.properties.is_empty() {
                    let json_obj: serde_json::Map<String, serde_json::Value> = scored
                        .entity
                        .properties
                        .iter()
                        .map(|(k, v)| (k.clone(), convert::value_to_json(v)))
                        .collect();
                    Some(async_graphql::Json(serde_json::Value::Object(json_obj)))
                } else {
                    None
                };

                VectorSearchResult {
                    id: ID(scored.entity.id.as_u64().to_string()),
                    score: scored.score as f64,
                    payload,
                }
            })
            .collect();

        Ok(search_results)
    }
}

/// Helper function to get collection info.
fn get_collection_info(db: &Database, name: &str) -> manifoldb::Result<CollectionInfo> {
    let handle = db.collection(name)?;
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
                    manifoldb::DistanceMetric::Cosine => DistanceMetricEnum::Cosine,
                    manifoldb::DistanceMetric::DotProduct => DistanceMetricEnum::DotProduct,
                    manifoldb::DistanceMetric::Euclidean => DistanceMetricEnum::Euclidean,
                    manifoldb::DistanceMetric::Manhattan => DistanceMetricEnum::Manhattan,
                    manifoldb::DistanceMetric::Chebyshev => DistanceMetricEnum::Chebyshev,
                },
                manifoldb::collection::DistanceType::Sparse(_) => DistanceMetricEnum::DotProduct,
                manifoldb::collection::DistanceType::Binary(_) => DistanceMetricEnum::Hamming,
            };

            VectorConfigInfo { name: vec_name.clone(), vector_type, dimension, distance_metric }
        })
        .collect();

    Ok(CollectionInfo {
        name: name.to_string(),
        vectors: vector_configs,
        point_count: point_count as i32,
    })
}
