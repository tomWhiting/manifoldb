//! GraphQL subscription resolvers.
//!
//! These resolvers handle real-time push notifications for graph changes.

use async_graphql::{Context, Result, Subscription, ID};
use futures_util::Stream;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use super::types::{
    EdgeCreatedEvent, EdgeDeletedEvent, EdgeEvent, EdgeUpdatedEvent, GraphEvent, NodeCreatedEvent,
    NodeDeletedEvent, NodeEvent, NodeUpdatedEvent,
};
use crate::pubsub::{EdgeChangeEvent, GraphChangeEvent, NodeChangeEvent, PubSub};

/// Root subscription type for the GraphQL schema.
pub struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    /// Subscribe to node changes.
    ///
    /// Optionally filter by labels - only events for nodes with at least one
    /// of the specified labels will be emitted.
    async fn node_changes(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Filter by node labels (any match)")] labels: Option<Vec<String>>,
    ) -> Result<impl Stream<Item = NodeEvent>> {
        let pubsub = ctx.data::<PubSub>()?;
        let receiver = pubsub.subscribe();

        let stream = BroadcastStream::new(receiver).filter_map(move |result| {
            let labels_filter = labels.clone();
            match result {
                Ok(GraphChangeEvent::Node(event)) => {
                    // Apply label filter if specified
                    if let Some(ref filter_labels) = labels_filter {
                        let event_labels = event.labels();
                        let matches = filter_labels.iter().any(|l| event_labels.contains(l));
                        if !matches {
                            return None;
                        }
                    }

                    // Convert internal event to GraphQL event
                    Some(convert_node_event(event))
                }
                Ok(GraphChangeEvent::Edge(_)) => None,
                Err(_) => None, // Lagged, skip
            }
        });

        Ok(stream)
    }

    /// Subscribe to edge changes.
    ///
    /// Optionally filter by edge types - only events for edges with one
    /// of the specified types will be emitted.
    async fn edge_changes(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Filter by edge types (any match)")] types: Option<Vec<String>>,
    ) -> Result<impl Stream<Item = EdgeEvent>> {
        let pubsub = ctx.data::<PubSub>()?;
        let receiver = pubsub.subscribe();

        let stream = BroadcastStream::new(receiver).filter_map(move |result| {
            let types_filter = types.clone();
            match result {
                Ok(GraphChangeEvent::Edge(event)) => {
                    // Apply type filter if specified
                    if let Some(ref filter_types) = types_filter {
                        let event_type = event.edge_type();
                        if !filter_types.iter().any(|t| t == event_type) {
                            return None;
                        }
                    }

                    // Convert internal event to GraphQL event
                    Some(convert_edge_event(event))
                }
                Ok(GraphChangeEvent::Node(_)) => None,
                Err(_) => None, // Lagged, skip
            }
        });

        Ok(stream)
    }

    /// Subscribe to all graph changes (nodes and edges).
    async fn graph_changes(&self, ctx: &Context<'_>) -> Result<impl Stream<Item = GraphEvent>> {
        let pubsub = ctx.data::<PubSub>()?;
        let receiver = pubsub.subscribe();

        let stream = BroadcastStream::new(receiver).filter_map(|result| match result {
            Ok(event) => Some(convert_graph_event(event)),
            Err(_) => None, // Lagged, skip
        });

        Ok(stream)
    }
}

/// Convert an internal node change event to a GraphQL node event.
fn convert_node_event(event: NodeChangeEvent) -> NodeEvent {
    match event {
        NodeChangeEvent::Created(node) => NodeEvent::Created(NodeCreatedEvent { node }),
        NodeChangeEvent::Updated(node) => NodeEvent::Updated(NodeUpdatedEvent { node }),
        NodeChangeEvent::Deleted { id, labels } => {
            NodeEvent::Deleted(NodeDeletedEvent { id: ID(id), labels })
        }
    }
}

/// Convert an internal edge change event to a GraphQL edge event.
fn convert_edge_event(event: EdgeChangeEvent) -> EdgeEvent {
    match event {
        EdgeChangeEvent::Created(edge) => EdgeEvent::Created(EdgeCreatedEvent { edge }),
        EdgeChangeEvent::Updated(edge) => EdgeEvent::Updated(EdgeUpdatedEvent { edge }),
        EdgeChangeEvent::Deleted { id, edge_type } => {
            EdgeEvent::Deleted(EdgeDeletedEvent { id: ID(id), edge_type })
        }
    }
}

/// Convert an internal graph change event to a GraphQL graph event.
fn convert_graph_event(event: GraphChangeEvent) -> GraphEvent {
    match event {
        GraphChangeEvent::Node(NodeChangeEvent::Created(node)) => {
            GraphEvent::NodeCreated(NodeCreatedEvent { node })
        }
        GraphChangeEvent::Node(NodeChangeEvent::Updated(node)) => {
            GraphEvent::NodeUpdated(NodeUpdatedEvent { node })
        }
        GraphChangeEvent::Node(NodeChangeEvent::Deleted { id, labels }) => {
            GraphEvent::NodeDeleted(NodeDeletedEvent { id: ID(id), labels })
        }
        GraphChangeEvent::Edge(EdgeChangeEvent::Created(edge)) => {
            GraphEvent::EdgeCreated(EdgeCreatedEvent { edge })
        }
        GraphChangeEvent::Edge(EdgeChangeEvent::Updated(edge)) => {
            GraphEvent::EdgeUpdated(EdgeUpdatedEvent { edge })
        }
        GraphChangeEvent::Edge(EdgeChangeEvent::Deleted { id, edge_type }) => {
            GraphEvent::EdgeDeleted(EdgeDeletedEvent { id: ID(id), edge_type })
        }
    }
}
