//! Pub-sub infrastructure for GraphQL subscriptions.
//!
//! This module provides a broadcast channel for publishing database change events
//! that can be consumed by GraphQL subscriptions.

use tokio::sync::broadcast;

use crate::schema::types::{Edge, Node};

/// The capacity of the broadcast channel.
/// Subscribers that fall behind by more than this many messages will miss events.
const CHANNEL_CAPACITY: usize = 256;

/// Events related to node changes.
#[derive(Clone, Debug)]
pub enum NodeChangeEvent {
    /// A node was created.
    Created(Node),
    /// A node was updated.
    Updated(Node),
    /// A node was deleted.
    Deleted {
        /// The ID of the deleted node.
        id: String,
        /// The labels the node had before deletion.
        labels: Vec<String>,
    },
}

impl NodeChangeEvent {
    /// Get the labels associated with this event's node.
    pub fn labels(&self) -> &[String] {
        match self {
            NodeChangeEvent::Created(node) | NodeChangeEvent::Updated(node) => &node.labels,
            NodeChangeEvent::Deleted { labels, .. } => labels,
        }
    }
}

/// Events related to edge changes.
#[derive(Clone, Debug)]
pub enum EdgeChangeEvent {
    /// An edge was created.
    Created(Edge),
    /// An edge was updated.
    Updated(Edge),
    /// An edge was deleted.
    Deleted {
        /// The ID of the deleted edge.
        id: String,
        /// The type the edge had before deletion.
        edge_type: String,
    },
}

impl EdgeChangeEvent {
    /// Get the edge type associated with this event.
    pub fn edge_type(&self) -> &str {
        match self {
            EdgeChangeEvent::Created(edge) | EdgeChangeEvent::Updated(edge) => &edge.edge_type,
            EdgeChangeEvent::Deleted { edge_type, .. } => edge_type,
        }
    }
}

/// A unified graph change event that can be either a node or edge change.
#[derive(Clone, Debug)]
pub enum GraphChangeEvent {
    /// A node change event.
    Node(NodeChangeEvent),
    /// An edge change event.
    Edge(EdgeChangeEvent),
}

/// Pub-sub hub for broadcasting graph change events.
///
/// This is stored in the GraphQL context and used by mutations to publish events
/// and by subscriptions to receive them.
#[derive(Clone)]
pub struct PubSub {
    /// Sender for graph change events.
    sender: broadcast::Sender<GraphChangeEvent>,
}

impl PubSub {
    /// Create a new pub-sub hub.
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self { sender }
    }

    /// Publish a node change event.
    pub fn publish_node_event(&self, event: NodeChangeEvent) {
        // Ignore send errors - they just mean there are no subscribers
        let _ = self.sender.send(GraphChangeEvent::Node(event));
    }

    /// Publish an edge change event.
    pub fn publish_edge_event(&self, event: EdgeChangeEvent) {
        // Ignore send errors - they just mean there are no subscribers
        let _ = self.sender.send(GraphChangeEvent::Edge(event));
    }

    /// Subscribe to all graph change events.
    pub fn subscribe(&self) -> broadcast::Receiver<GraphChangeEvent> {
        self.sender.subscribe()
    }
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PubSub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PubSub")
            .field("subscriber_count", &self.sender.receiver_count())
            .finish()
    }
}
