//! GraphQL type definitions.
//!
//! These types represent the data structures exposed through the GraphQL API.

use async_graphql::{Enum, InputObject, Json, SimpleObject, Union, ID};

/// A node in the graph.
#[derive(SimpleObject, Clone, Debug)]
pub struct Node {
    /// Unique identifier for the node.
    pub id: ID,
    /// Labels attached to this node.
    pub labels: Vec<String>,
    /// Properties as a JSON object.
    pub properties: Json<serde_json::Value>,
}

/// An edge in the graph.
#[derive(SimpleObject, Clone, Debug)]
pub struct Edge {
    /// Unique identifier for the edge.
    pub id: ID,
    /// The type/label of the relationship.
    pub edge_type: String,
    /// Source node ID.
    pub source: ID,
    /// Target node ID.
    pub target: ID,
    /// Properties as a JSON object.
    pub properties: Json<serde_json::Value>,
}

/// Result of a Cypher/graph query.
#[derive(SimpleObject, Clone, Debug, Default)]
pub struct GraphResult {
    /// Nodes in the result.
    pub nodes: Vec<Node>,
    /// Edges in the result.
    pub edges: Vec<Edge>,
}

/// Result of a SQL/tabular query.
#[derive(SimpleObject, Clone, Debug)]
pub struct TableResult {
    /// Column names.
    pub columns: Vec<String>,
    /// Rows as arrays of JSON values.
    pub rows: Vec<Json<Vec<serde_json::Value>>>,
    /// Total row count.
    pub row_count: i32,
}

/// Combined result that can hold either tabular or graph format.
#[derive(SimpleObject, Clone, Debug)]
pub struct QueryResult {
    /// Tabular result (for SQL queries).
    pub table: Option<TableResult>,
    /// Graph result (for Cypher queries).
    pub graph: Option<GraphResult>,
}

/// Information about a node label.
#[derive(SimpleObject, Clone, Debug)]
pub struct LabelInfo {
    /// The label name.
    pub name: String,
    /// Number of nodes with this label.
    pub count: i64,
}

/// Information about an edge type.
#[derive(SimpleObject, Clone, Debug)]
pub struct EdgeTypeInfo {
    /// The edge type name.
    pub name: String,
    /// Number of edges of this type.
    pub count: i64,
}

/// Database graph statistics.
#[derive(SimpleObject, Clone, Debug)]
pub struct GraphStats {
    /// Total number of nodes.
    pub node_count: i64,
    /// Total number of edges.
    pub edge_count: i64,
    /// Breakdown of nodes by label.
    pub labels: Vec<LabelInfo>,
    /// Breakdown of edges by type.
    pub edge_types: Vec<EdgeTypeInfo>,
}

/// Direction for graph traversal.
#[derive(Enum, Copy, Clone, Eq, PartialEq, Debug)]
pub enum Direction {
    /// Outgoing edges only.
    Outgoing,
    /// Incoming edges only.
    Incoming,
    /// Both directions.
    Both,
}

/// Input for creating a new node.
#[derive(InputObject, Debug)]
pub struct CreateNodeInput {
    /// Labels to attach to the node.
    pub labels: Vec<String>,
    /// Properties as a JSON object.
    pub properties: Option<Json<serde_json::Value>>,
}

/// Input for creating a new edge.
#[derive(InputObject, Debug)]
pub struct CreateEdgeInput {
    /// Source node ID.
    pub source_id: String,
    /// Target node ID.
    pub target_id: String,
    /// Edge type/label.
    pub edge_type: String,
    /// Properties as a JSON object.
    pub properties: Option<Json<serde_json::Value>>,
}

// =============================================================================
// Subscription Event Types
// =============================================================================

/// Event emitted when a node is created.
#[derive(SimpleObject, Clone, Debug)]
pub struct NodeCreatedEvent {
    /// The created node.
    pub node: Node,
}

/// Event emitted when a node is updated.
#[derive(SimpleObject, Clone, Debug)]
pub struct NodeUpdatedEvent {
    /// The updated node.
    pub node: Node,
}

/// Event emitted when a node is deleted.
#[derive(SimpleObject, Clone, Debug)]
pub struct NodeDeletedEvent {
    /// The ID of the deleted node.
    pub id: ID,
    /// The labels the node had before deletion.
    pub labels: Vec<String>,
}

/// A node change event (created, updated, or deleted).
#[derive(Union, Clone, Debug)]
pub enum NodeEvent {
    /// A node was created.
    Created(NodeCreatedEvent),
    /// A node was updated.
    Updated(NodeUpdatedEvent),
    /// A node was deleted.
    Deleted(NodeDeletedEvent),
}

/// Event emitted when an edge is created.
#[derive(SimpleObject, Clone, Debug)]
pub struct EdgeCreatedEvent {
    /// The created edge.
    pub edge: Edge,
}

/// Event emitted when an edge is updated.
#[derive(SimpleObject, Clone, Debug)]
pub struct EdgeUpdatedEvent {
    /// The updated edge.
    pub edge: Edge,
}

/// Event emitted when an edge is deleted.
#[derive(SimpleObject, Clone, Debug)]
pub struct EdgeDeletedEvent {
    /// The ID of the deleted edge.
    pub id: ID,
    /// The type of the deleted edge.
    pub edge_type: String,
}

/// An edge change event (created, updated, or deleted).
#[derive(Union, Clone, Debug)]
pub enum EdgeEvent {
    /// An edge was created.
    Created(EdgeCreatedEvent),
    /// An edge was updated.
    Updated(EdgeUpdatedEvent),
    /// An edge was deleted.
    Deleted(EdgeDeletedEvent),
}

/// A graph change event (any node or edge change).
#[derive(Union, Clone, Debug)]
pub enum GraphEvent {
    /// A node was created.
    NodeCreated(NodeCreatedEvent),
    /// A node was updated.
    NodeUpdated(NodeUpdatedEvent),
    /// A node was deleted.
    NodeDeleted(NodeDeletedEvent),
    /// An edge was created.
    EdgeCreated(EdgeCreatedEvent),
    /// An edge was updated.
    EdgeUpdated(EdgeUpdatedEvent),
    /// An edge was deleted.
    EdgeDeleted(EdgeDeletedEvent),
}
