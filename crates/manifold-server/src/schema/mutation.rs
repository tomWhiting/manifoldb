//! GraphQL mutation resolvers.
//!
//! These resolvers handle write operations against the database.

use std::sync::Arc;

use async_graphql::{Context, Object, Result, ID};
use manifoldb::Database;

use super::types::{CreateEdgeInput, CreateNodeInput, Edge, Node, QueryResult};
use crate::convert;
use crate::pubsub::{EdgeChangeEvent, NodeChangeEvent, PubSub};

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
        let pubsub = ctx.data::<PubSub>()?;

        let labels_str = input.labels.join(":");
        let props = input
            .properties
            .map(|p| format!(" {}", p.0))
            .unwrap_or_default();

        let query = format!("CREATE (n:{}{}) RETURN n", labels_str, props);
        let result = db.query(&query)?;

        let graph = convert::query_result_to_graph(&result)?;
        let node = graph
            .nodes
            .into_iter()
            .next()
            .ok_or_else(|| async_graphql::Error::new("Failed to create node"))?;

        // Publish event
        pubsub.publish_node_event(NodeChangeEvent::Created(node.clone()));

        Ok(node)
    }

    /// Create an edge between two nodes.
    async fn create_edge(&self, ctx: &Context<'_>, input: CreateEdgeInput) -> Result<Edge> {
        let db = ctx.data::<Arc<Database>>()?;
        let pubsub = ctx.data::<PubSub>()?;

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
        let edge = graph
            .edges
            .into_iter()
            .next()
            .ok_or_else(|| async_graphql::Error::new("Failed to create edge"))?;

        // Publish event
        pubsub.publish_edge_event(EdgeChangeEvent::Created(edge.clone()));

        Ok(edge)
    }

    /// Delete a node by ID (and its connected edges).
    async fn delete_node(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let db = ctx.data::<Arc<Database>>()?;
        let pubsub = ctx.data::<PubSub>()?;

        // First, get the node info before deletion
        let node_query = format!("MATCH (n) WHERE id(n) = {} RETURN n", id.as_str());
        let node_result = db.query(&node_query)?;
        let node_graph = convert::query_result_to_graph(&node_result)?;
        let node_info = node_graph.nodes.into_iter().next();

        // Also get connected edges before deletion
        let edges_query = format!(
            "MATCH (n)-[r]-() WHERE id(n) = {} RETURN r",
            id.as_str()
        );
        let edges_result = db.query(&edges_query)?;
        let edges_graph = convert::query_result_to_graph(&edges_result)?;

        // Delete the node
        let query = format!("MATCH (n) WHERE id(n) = {} DETACH DELETE n", id.as_str());
        db.execute(&query)?;

        // Publish edge deleted events for all connected edges
        for edge in edges_graph.edges {
            pubsub.publish_edge_event(EdgeChangeEvent::Deleted {
                id: edge.id.to_string(),
                edge_type: edge.edge_type,
            });
        }

        // Publish node deleted event
        if let Some(node) = node_info {
            pubsub.publish_node_event(NodeChangeEvent::Deleted {
                id: node.id.to_string(),
                labels: node.labels,
            });
        }

        Ok(true)
    }

    /// Delete an edge by ID.
    async fn delete_edge(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let db = ctx.data::<Arc<Database>>()?;
        let pubsub = ctx.data::<PubSub>()?;

        // First, get the edge info before deletion
        let edge_query = format!("MATCH ()-[r]->() WHERE id(r) = {} RETURN r", id.as_str());
        let edge_result = db.query(&edge_query)?;
        let edge_graph = convert::query_result_to_graph(&edge_result)?;
        let edge_info = edge_graph.edges.into_iter().next();

        // Delete the edge
        let query = format!("MATCH ()-[r]->() WHERE id(r) = {} DELETE r", id.as_str());
        db.execute(&query)?;

        // Publish edge deleted event
        if let Some(edge) = edge_info {
            pubsub.publish_edge_event(EdgeChangeEvent::Deleted {
                id: edge.id.to_string(),
                edge_type: edge.edge_type,
            });
        }

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
        let pubsub = ctx.data::<PubSub>()?;

        let query = format!(
            "MATCH (n) WHERE id(n) = {} SET n += {} RETURN n",
            id.as_str(),
            properties.0
        );

        let result = db.query(&query)?;

        let graph = convert::query_result_to_graph(&result)?;
        let node = graph
            .nodes
            .into_iter()
            .next()
            .ok_or_else(|| async_graphql::Error::new("Node not found"))?;

        // Publish event
        pubsub.publish_node_event(NodeChangeEvent::Updated(node.clone()));

        Ok(node)
    }
}
