//! Property-based tests for encoding round-trips.

#![allow(clippy::expect_used, clippy::float_cmp)]

use proptest::prelude::*;

use crate::encoding::value::decode_value;
use crate::encoding::{Decoder, Encoder};
use crate::types::{Edge, EdgeId, EdgeType, Entity, EntityId, Label, Value};

/// Strategy for generating arbitrary `Value` instances.
fn arb_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::Int),
        // Filter out NaN since NaN != NaN
        any::<f64>().prop_filter("not NaN", |f| !f.is_nan()).prop_map(Value::Float),
        ".*".prop_map(Value::String),
        prop::collection::vec(any::<u8>(), 0..100).prop_map(Value::Bytes),
        prop::collection::vec(any::<f32>().prop_filter("not NaN", |f| !f.is_nan()), 0..50)
            .prop_map(Value::Vector),
    ];

    leaf.prop_recursive(
        3,  // depth
        64, // size
        10, // items per collection
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
            let mut edge =
                Edge::new(EdgeId::new(id), EntityId::new(source), EntityId::new(target), edge_type);
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

    /// Corrupted/arbitrary bytes should not crash, only return errors.
    #[test]
    fn arbitrary_bytes_dont_crash(bytes in prop::collection::vec(any::<u8>(), 0..1000)) {
        // This should either succeed or return an error, never panic
        let _ = Value::decode(&bytes);
    }

    /// Truncated valid encodings should return errors, not panic.
    #[test]
    fn truncated_encoding_returns_error(value in arb_value()) {
        let encoded = value.encode().expect("encoding should succeed");
        if encoded.len() > 1 {
            // Try all possible truncations
            for truncate_at in 1..encoded.len() {
                let truncated = &encoded[..truncate_at];
                // Should either succeed (if truncated is a valid prefix) or return error
                let _ = Value::decode(truncated);
            }
        }
    }

    /// Mutated encodings should return errors or valid values, never panic.
    #[test]
    fn mutated_encoding_returns_error_or_value(
        value in arb_value(),
        mutation_idx in any::<usize>(),
        mutation_val in any::<u8>()
    ) {
        let mut encoded = value.encode().expect("encoding should succeed");
        if !encoded.is_empty() {
            let idx = mutation_idx % encoded.len();
            encoded[idx] = mutation_val;
            // Should either succeed or return error, never panic
            let _ = Value::decode(&encoded);
        }
    }

    /// Test that decode_value correctly reports consumed bytes.
    #[test]
    fn decode_value_reports_correct_consumed(value in arb_value()) {
        let encoded = value.encode().expect("encoding should succeed");
        let (decoded, consumed) = decode_value(&encoded).expect("decoding should succeed");
        prop_assert_eq!(value, decoded);
        prop_assert_eq!(encoded.len(), consumed);
    }

    /// Large length headers shouldn't cause allocation panics.
    #[test]
    fn large_length_header_doesnt_panic(tag in 4u8..=9u8, len_bytes in any::<[u8; 4]>()) {
        // Create bytes with a type tag followed by a potentially large length
        let mut bytes = vec![tag];
        bytes.extend_from_slice(&len_bytes);
        // Add a small amount of trailing data
        bytes.extend_from_slice(&[0u8; 16]);
        // Should return an error for truncated data, not panic from OOM
        let _ = Value::decode(&bytes);
    }
}
