//! GraphQL query resolvers.
//!
//! These resolvers handle read operations against the database.

use std::sync::Arc;

use async_graphql::{Context, Object, Result, ID};
use manifoldb::Database;

use super::types::{Direction, EdgeTypeInfo, GraphResult, GraphStats, LabelInfo, Node, TableResult};
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
                .map(|(name, count)| LabelInfo {
                    name,
                    count: count as i64,
                })
                .collect(),
            edge_types: stats
                .edge_types
                .into_iter()
                .map(|(name, count)| EdgeTypeInfo {
                    name,
                    count: count as i64,
                })
                .collect(),
        })
    }

    /// Get all node labels with counts.
    async fn labels(&self, ctx: &Context<'_>) -> Result<Vec<LabelInfo>> {
        let db = ctx.data::<Arc<Database>>()?;
        let labels = db.list_labels()?;
        Ok(labels
            .into_iter()
            .map(|(name, count)| LabelInfo {
                name,
                count: count as i64,
            })
            .collect())
    }

    /// Get all edge types with counts.
    async fn edge_types(&self, ctx: &Context<'_>) -> Result<Vec<EdgeTypeInfo>> {
        let db = ctx.data::<Arc<Database>>()?;
        let edge_types = db.list_edge_types()?;
        Ok(edge_types
            .into_iter()
            .map(|(name, count)| EdgeTypeInfo {
                name,
                count: count as i64,
            })
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

        let type_filter = edge_type
            .map(|t| format!(":{}", t))
            .unwrap_or_default();

        let (left_arrow, right_arrow) = match direction.unwrap_or(Direction::Both) {
            Direction::Outgoing => ("", "->"),
            Direction::Incoming => ("<-", ""),
            Direction::Both => ("", "-"),
        };

        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();

        let query = format!(
            "MATCH (n){}[r{}]{}(m) WHERE id(n) = {} RETURN n, r, m{}",
            left_arrow, type_filter, right_arrow, node_id.as_str(), limit_clause
        );

        let result = db.query(&query)?;
        convert::query_result_to_graph(&result)
    }
}
