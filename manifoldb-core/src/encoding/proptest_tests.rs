//! Property-based tests for encoding round-trips.

#![allow(clippy::expect_used, clippy::float_cmp)]

use proptest::prelude::*;

use crate::encoding::{Decoder, Encoder};
use crate::types::{Edge, EdgeId, EdgeType, Entity, EntityId, Label, Value};

/// Strategy for generating arbitrary `Value` instances.
fn arb_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::Int),
        // Filter out NaN since NaN != NaN
        any::<f64>()
            .prop_filter("not NaN", |f| !f.is_nan())
            .prop_map(Value::Float),
        ".*".prop_map(Value::String),
        prop::collection::vec(any::<u8>(), 0..100).prop_map(Value::Bytes),
        prop::collection::vec(
            any::<f32>().prop_filter("not NaN", |f| !f.is_nan()),
            0..50
        )
        .prop_map(Value::Vector),
    ];

    leaf.prop_recursive(
        3,   // depth
        64,  // size
        10,  // items per collection
        |inner| prop::collection::vec(inner, 0..10).prop_map(Value::Array),
    )
}

/// Strategy for generating arbitrary `Label` instances.
fn arb_label() -> impl Strategy<Value = Label> {
    "[a-zA-Z][a-zA-Z0-9_]*".prop_map(Label::new)
}

/// Strategy for generating arbitrary `Entity` instances.
fn arb_entity() -> impl Strategy<Value = Entity> {
    (
        any::<u64>(),
        prop::collection::vec(arb_label(), 0..5),
        prop::collection::hash_map("[a-zA-Z_][a-zA-Z0-9_]*", arb_value(), 0..10),
    )
        .prop_map(|(id, labels, properties)| {
            let mut entity = Entity::new(EntityId::new(id));
            entity.labels = labels;
            entity.properties = properties;
            entity
        })
}

/// Strategy for generating arbitrary `EdgeType` instances.
fn arb_edge_type() -> impl Strategy<Value = EdgeType> {
    "[A-Z][A-Z_]*".prop_map(EdgeType::new)
}

/// Strategy for generating arbitrary `Edge` instances.
fn arb_edge() -> impl Strategy<Value = Edge> {
    (
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        arb_edge_type(),
        prop::collection::hash_map("[a-zA-Z_][a-zA-Z0-9_]*", arb_value(), 0..10),
    )
        .prop_map(|(id, source, target, edge_type, properties)| {
            let mut edge = Edge::new(
                EdgeId::new(id),
                EntityId::new(source),
                EntityId::new(target),
                edge_type,
            );
            edge.properties = properties;
            edge
        })
}

proptest! {
    #[test]
    fn value_roundtrip(value in arb_value()) {
        let encoded = value.encode().expect("encoding should succeed");
        let decoded = Value::decode(&encoded).expect("decoding should succeed");
        prop_assert_eq!(value, decoded);
    }

    #[test]
    fn entity_roundtrip(entity in arb_entity()) {
        let encoded = entity.encode().expect("encoding should succeed");
        let decoded = Entity::decode(&encoded).expect("decoding should succeed");
        prop_assert_eq!(entity.id, decoded.id);
        prop_assert_eq!(entity.labels.len(), decoded.labels.len());
        prop_assert_eq!(entity.properties, decoded.properties);
    }

    #[test]
    fn edge_roundtrip(edge in arb_edge()) {
        let encoded = edge.encode().expect("encoding should succeed");
        let decoded = Edge::decode(&encoded).expect("decoding should succeed");
        prop_assert_eq!(edge.id, decoded.id);
        prop_assert_eq!(edge.source, decoded.source);
        prop_assert_eq!(edge.target, decoded.target);
        prop_assert_eq!(edge.edge_type.as_str(), decoded.edge_type.as_str());
        prop_assert_eq!(edge.properties, decoded.properties);
    }

    #[test]
    fn int_value_preserves_bits(i in any::<i64>()) {
        let value = Value::Int(i);
        let encoded = value.encode().expect("encoding should succeed");
        let decoded = Value::decode(&encoded).expect("decoding should succeed");
        match decoded {
            Value::Int(decoded_i) => prop_assert_eq!(i, decoded_i),
            _ => prop_assert!(false, "expected Int variant"),
        }
    }

    #[test]
    fn float_value_preserves_bits(f in any::<f64>().prop_filter("not NaN", |f| !f.is_nan())) {
        let value = Value::Float(f);
        let encoded = value.encode().expect("encoding should succeed");
        let decoded = Value::decode(&encoded).expect("decoding should succeed");
        match decoded {
            Value::Float(decoded_f) => prop_assert_eq!(f, decoded_f),
            _ => prop_assert!(false, "expected Float variant"),
        }
    }

    #[test]
    fn vector_value_preserves_all_elements(v in prop::collection::vec(
        any::<f32>().prop_filter("not NaN", |f| !f.is_nan()),
        0..100
    )) {
        let value = Value::Vector(v.clone());
        let encoded = value.encode().expect("encoding should succeed");
        let decoded = Value::decode(&encoded).expect("decoding should succeed");
        match decoded {
            Value::Vector(decoded_v) => {
                prop_assert_eq!(v.len(), decoded_v.len());
                for (a, b) in v.iter().zip(decoded_v.iter()) {
                    prop_assert_eq!(a, b);
                }
            }
            _ => prop_assert!(false, "expected Vector variant"),
        }
    }

    #[test]
    fn string_value_roundtrip(s in ".*") {
        let value = Value::String(s.clone());
        let encoded = value.encode().expect("encoding should succeed");
        let decoded = Value::decode(&encoded).expect("decoding should succeed");
        match decoded {
            Value::String(decoded_s) => prop_assert_eq!(s, decoded_s),
            _ => prop_assert!(false, "expected String variant"),
        }
    }

    #[test]
    fn entity_id_roundtrip_via_entity(id in any::<u64>()) {
        let entity = Entity::new(EntityId::new(id));
        let encoded = entity.encode().expect("encoding should succeed");
        let decoded = Entity::decode(&encoded).expect("decoding should succeed");
        prop_assert_eq!(id, decoded.id.as_u64());
    }
}
