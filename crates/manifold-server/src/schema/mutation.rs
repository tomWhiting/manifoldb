//! GraphQL mutation resolvers.
//!
//! These resolvers handle write operations against the database.

use std::sync::Arc;

use async_graphql::{Context, Object, Result, ID};
use manifoldb::Database;

use super::types::{CreateEdgeInput, CreateNodeInput, Edge, Node, QueryResult};
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
}
