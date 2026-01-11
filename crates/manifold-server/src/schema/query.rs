//! GraphQL query resolvers.
//!
//! These resolvers handle read operations against the database.

use std::sync::Arc;

use async_graphql::{Context, Object, Result, ID};
use manifoldb::collection::{CollectionManager, CollectionName};
use manifoldb::Database;
use manifoldb_vector::types::Embedding;

use super::types::{
    CollectionInfo, Direction, EdgeTypeInfo, EmbeddingModelInfo, EmbeddingResult, GraphResult,
    GraphStats, LabelInfo, Node, NodeVector, TableResult, TextSearchInput, VectorConfigInfo,
    VectorSearchInput, VectorSearchResult,
};
use crate::convert;
use crate::embedding::EmbeddingService;

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
        use manifoldb::vector::search_named_vector_index;

        let db = ctx.data::<Arc<Database>>()?;

        // Convert f64 query vector to f32
        let query_vector: Vec<f32> = input.query_vector.iter().map(|v| *v as f32).collect();

        let tx = db.begin_read()?;

        // Create embedding for search
        let query_embedding = Embedding::new(query_vector)
            .map_err(|e| async_graphql::Error::new(format!("Invalid query vector: {}", e)))?;

        let limit = input.limit.unwrap_or(10) as usize;
        let offset = input.offset.unwrap_or(0) as usize;

        // Search using the HNSW index
        let results = search_named_vector_index(
            &tx,
            &collection,
            &input.vector_name,
            &query_embedding,
            limit + offset, // Fetch extra for offset
            None,
        )
        .map_err(|e| async_graphql::Error::new(format!("Search failed: {}", e)))?;

        // Get collection ID for payload lookup
        let coll_name = CollectionName::new(&collection)
            .map_err(|e| async_graphql::Error::new(format!("Invalid collection name: {}", e)))?;
        // Verify collection exists
        CollectionManager::get(&tx, &coll_name)
            .map_err(|e| async_graphql::Error::new(format!("Failed to get collection: {}", e)))?
            .ok_or_else(|| {
                async_graphql::Error::new(format!("Collection '{}' not found", collection))
            })?;

        // Apply offset and convert to GraphQL results
        let threshold = input.score_threshold.map(|t| t as f32);

        let search_results: Vec<VectorSearchResult> = results
            .into_iter()
            .skip(offset)
            .take(limit)
            .filter(|r| threshold.is_none_or(|t| r.distance <= t))
            .map(|result| {
                // Convert distance to score (lower distance = higher score)
                let score = 1.0 / (1.0 + result.distance as f64);

                // Payloads are not stored with vectors in the current implementation
                // Users can store metadata via entity properties if needed
                VectorSearchResult {
                    id: ID(result.entity_id.as_u64().to_string()),
                    score,
                    payload: None,
                }
            })
            .collect();

        Ok(search_results)
    }

    /// Get vectors for specific node IDs.
    ///
    /// This query retrieves the raw vector data stored for the specified nodes.
    /// If collection and vector_name are provided, only vectors from that specific
    /// collection and vector field are returned. Otherwise, all vectors for the
    /// nodes are returned.
    async fn node_vectors(
        &self,
        ctx: &Context<'_>,
        node_ids: Vec<ID>,
        collection: Option<String>,
        vector_name: Option<String>,
    ) -> Result<Vec<NodeVector>> {
        use manifoldb_core::EntityId;
        use manifoldb_storage::{Cursor, Transaction as StorageTransaction};
        use manifoldb_vector::encoding::{
            decode_collection_vector_key, encode_entity_vector_prefix,
        };
        use manifoldb_vector::store::decode_vector_value;
        use manifoldb_vector::TABLE_COLLECTION_VECTORS;
        use std::ops::Bound;

        let db = ctx.data::<Arc<Database>>()?;
        let tx = db.begin_read()?;

        let mut results = Vec::new();

        // If collection is specified, get the collection ID
        let collection_info = if let Some(ref coll_name) = collection {
            let name = CollectionName::new(coll_name).map_err(|e| {
                async_graphql::Error::new(format!("Invalid collection name: {}", e))
            })?;
            let coll = CollectionManager::get(&tx, &name)
                .map_err(|e| async_graphql::Error::new(format!("Failed to get collection: {}", e)))?
                .ok_or_else(|| {
                    async_graphql::Error::new(format!("Collection '{}' not found", coll_name))
                })?;
            Some((coll.id(), coll_name.clone()))
        } else {
            None
        };

        // Get storage reference for range scans
        let storage = tx
            .storage_ref()
            .map_err(|e| async_graphql::Error::new(format!("Storage error: {}", e)))?;

        // Process each node ID
        for node_id in node_ids {
            let entity_id = EntityId::new(node_id.as_str().parse::<u64>().map_err(|_| {
                async_graphql::Error::new(format!("Invalid node ID: {}", node_id.as_str()))
            })?);

            if let Some((collection_id, ref coll_name)) = collection_info {
                // Get vectors for specific collection
                let prefix = encode_entity_vector_prefix(collection_id, entity_id);
                let prefix_end = {
                    let mut end = prefix.clone();
                    for byte in end.iter_mut().rev() {
                        if *byte < 0xFF {
                            *byte += 1;
                            break;
                        }
                    }
                    end
                };

                let mut cursor = storage
                    .range(
                        TABLE_COLLECTION_VECTORS,
                        Bound::Included(prefix.as_slice()),
                        Bound::Excluded(prefix_end.as_slice()),
                    )
                    .map_err(|e| async_graphql::Error::new(format!("Storage error: {}", e)))?;

                while let Some((key, value)) = cursor
                    .next()
                    .map_err(|e| async_graphql::Error::new(format!("Cursor error: {}", e)))?
                {
                    if let Some(decoded_key) = decode_collection_vector_key(&key) {
                        // If vector_name is specified, filter by it
                        if let Some(ref vn) = vector_name {
                            let hash = manifoldb_vector::encoding::hash_name(vn);
                            if decoded_key.vector_name_hash != hash {
                                continue;
                            }
                        }

                        if let Ok((vector_data, vec_name)) = decode_vector_value(&value) {
                            // Filter by vector_name if specified
                            if let Some(ref vn) = vector_name {
                                if vec_name != *vn {
                                    continue;
                                }
                            }

                            if let Some(dense) = vector_data.as_dense() {
                                results.push(NodeVector {
                                    node_id: node_id.clone(),
                                    collection: Some(coll_name.clone()),
                                    vector_name: Some(vec_name),
                                    values: dense.iter().map(|v| *v as f64).collect(),
                                    dimension: dense.len() as i32,
                                });
                            }
                        }
                    }
                }
            } else {
                // No collection specified - scan all collections for this entity
                // This is more expensive but provides a complete picture
                let collection_names = db.list_collections().map_err(|e| {
                    async_graphql::Error::new(format!("Failed to list collections: {}", e))
                })?;

                for coll_name in collection_names {
                    if let Ok(name) = CollectionName::new(&coll_name) {
                        if let Ok(Some(coll)) = CollectionManager::get(&tx, &name) {
                            let collection_id = coll.id();
                            let prefix = encode_entity_vector_prefix(collection_id, entity_id);
                            let prefix_end = {
                                let mut end = prefix.clone();
                                for byte in end.iter_mut().rev() {
                                    if *byte < 0xFF {
                                        *byte += 1;
                                        break;
                                    }
                                }
                                end
                            };

                            if let Ok(mut cursor) = storage.range(
                                TABLE_COLLECTION_VECTORS,
                                Bound::Included(prefix.as_slice()),
                                Bound::Excluded(prefix_end.as_slice()),
                            ) {
                                while let Ok(Some((key, value))) = cursor.next() {
                                    if let Some(_decoded_key) = decode_collection_vector_key(&key) {
                                        if let Ok((vector_data, vec_name)) =
                                            decode_vector_value(&value)
                                        {
                                            // Filter by vector_name if specified
                                            if let Some(ref vn) = vector_name {
                                                if vec_name != *vn {
                                                    continue;
                                                }
                                            }

                                            if let Some(dense) = vector_data.as_dense() {
                                                results.push(NodeVector {
                                                    node_id: node_id.clone(),
                                                    collection: Some(coll_name.clone()),
                                                    vector_name: Some(vec_name),
                                                    values: dense
                                                        .iter()
                                                        .map(|v| *v as f64)
                                                        .collect(),
                                                    dimension: dense.len() as i32,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    // =========================================================================
    // Embedding Queries
    // =========================================================================

    /// List available embedding models.
    ///
    /// Returns information about all models that can be used for text embedding.
    async fn embedding_models(&self, ctx: &Context<'_>) -> Result<Vec<EmbeddingModelInfo>> {
        let service = ctx.data::<Arc<EmbeddingService>>()?;
        let models = service.list_models();

        Ok(models
            .into_iter()
            .map(|m| EmbeddingModelInfo {
                id: m.id.to_string(),
                name: m.name.to_string(),
                dimension: m.dimension,
                model_type: m.model_type.to_string(),
            })
            .collect())
    }

    /// Embed text into a vector using the specified model.
    ///
    /// The model is lazy-loaded on first use. If no model is specified,
    /// uses the server's default model (jina-embeddings-v2-small-en).
    async fn embed(
        &self,
        ctx: &Context<'_>,
        text: String,
        model: Option<String>,
    ) -> Result<EmbeddingResult> {
        let service = ctx.data::<Arc<EmbeddingService>>()?;
        let model_id = model.as_deref().unwrap_or_else(|| service.default_model());

        let embedding = service
            .encode(&text, model_id)
            .map_err(|e| async_graphql::Error::new(format!("Embedding failed: {}", e)))?;

        let dimension = embedding.len() as i32;

        Ok(EmbeddingResult {
            text,
            embedding: embedding.into_iter().map(|v| v as f64).collect(),
            dimension,
            model: model_id.to_string(),
        })
    }

    /// Search for similar vectors using text (embeds the query automatically).
    ///
    /// This is a convenience method that combines embedding and search.
    /// The query text is embedded using the model from the vector config,
    /// or the input override, or the server default.
    ///
    /// Model selection priority:
    /// 1. `input.model` - explicit user override
    /// 2. Vector config's `embedding_model` - model used when creating the collection
    /// 3. Server default model
    async fn search_by_text(
        &self,
        ctx: &Context<'_>,
        collection: String,
        input: TextSearchInput,
    ) -> Result<Vec<VectorSearchResult>> {
        use manifoldb::vector::search_named_vector_index;

        let db = ctx.data::<Arc<Database>>()?;
        let service = ctx.data::<Arc<EmbeddingService>>()?;

        let tx = db.begin_read()?;

        // Get collection to look up the vector config's embedding model
        let coll_name = CollectionName::new(&collection)
            .map_err(|e| async_graphql::Error::new(format!("Invalid collection name: {}", e)))?;
        let coll = CollectionManager::get(&tx, &coll_name)
            .map_err(|e| async_graphql::Error::new(format!("Failed to get collection: {}", e)))?
            .ok_or_else(|| {
                async_graphql::Error::new(format!("Collection '{}' not found", collection))
            })?;

        // Get the vector config's embedding model
        let vector_config = coll.vectors().get(&input.vector_name);
        let config_model = vector_config.and_then(|c| c.embedding_model.as_deref());

        // Determine which model to use: input override > config model > server default
        let model_id = input
            .model
            .as_deref()
            .or(config_model)
            .unwrap_or_else(|| service.default_model());

        // Embed the query text
        let query_vector = service
            .encode(&input.query_text, model_id)
            .map_err(|e| async_graphql::Error::new(format!("Embedding failed: {}", e)))?;

        // Create embedding for search
        let query_embedding = Embedding::new(query_vector)
            .map_err(|e| async_graphql::Error::new(format!("Invalid query vector: {}", e)))?;

        let limit = input.limit.unwrap_or(10) as usize;
        let offset = input.offset.unwrap_or(0) as usize;

        // Search using the HNSW index
        let results = search_named_vector_index(
            &tx,
            &collection,
            &input.vector_name,
            &query_embedding,
            limit + offset,
            None,
        )
        .map_err(|e| async_graphql::Error::new(format!("Search failed: {}", e)))?;

        // Apply offset and convert to GraphQL results
        let threshold = input.score_threshold.map(|t| t as f32);

        let search_results: Vec<VectorSearchResult> = results
            .into_iter()
            .skip(offset)
            .take(limit)
            .filter(|r| threshold.is_none_or(|t| r.distance <= t))
            .map(|result| {
                let score = 1.0 / (1.0 + result.distance as f64);
                VectorSearchResult {
                    id: ID(result.entity_id.as_u64().to_string()),
                    score,
                    payload: None,
                }
            })
            .collect();

        Ok(search_results)
    }
}

/// Helper function to get collection info.
fn get_collection_info(db: &Database, name: &str) -> manifoldb::Result<CollectionInfo> {
    use manifoldb_vector::{encoding::encode_collection_vector_prefix, TABLE_COLLECTION_VECTORS};
    use std::ops::Bound;

    let tx = db.begin_read()?;
    let coll_name =
        CollectionName::new(name).map_err(|e| manifoldb::Error::Collection(e.to_string()))?;
    let collection = CollectionManager::get(&tx, &coll_name)
        .map_err(|e| manifoldb::Error::Collection(e.to_string()))?
        .ok_or_else(|| manifoldb::Error::Collection(format!("Collection '{}' not found", name)))?;

    let vectors: Vec<VectorConfigInfo> = collection
        .vectors()
        .iter()
        .map(|(vec_name, config)| {
            let vector_type = convert::vector_type_to_graphql(&config.vector_type);
            VectorConfigInfo {
                name: vec_name.clone(),
                vector_type,
                dimension: config.dimension().map(|d| d as i32),
                distance_metric: convert::distance_to_graphql(&config.distance),
                embedding_model: config.embedding_model.clone(),
            }
        })
        .collect();

    // Count vectors in the collection by scanning the prefix
    let collection_id = collection.id();
    let prefix = encode_collection_vector_prefix(collection_id);
    let prefix_end = {
        let mut end = prefix.clone();
        for byte in end.iter_mut().rev() {
            if *byte < 0xFF {
                *byte += 1;
                break;
            }
        }
        end
    };

    let storage = tx.storage_ref().map_err(manifoldb::Error::Transaction)?;
    let mut cursor = {
        use manifoldb_storage::Transaction;
        storage
            .range(
                TABLE_COLLECTION_VECTORS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )
            .map_err(manifoldb::Error::Storage)?
    };

    let mut point_count = 0;
    while {
        use manifoldb_storage::Cursor;
        cursor.next().map_err(manifoldb::Error::Storage)?
    }
    .is_some()
    {
        point_count += 1;
    }

    Ok(CollectionInfo { name: name.to_string(), vectors, point_count })
}
