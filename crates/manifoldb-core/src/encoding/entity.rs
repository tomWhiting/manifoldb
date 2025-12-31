//! Serialization for [`Entity`] types.
//!
//! This module provides compact binary encoding for entities (nodes) in the graph.
//!
//! # Format
//!
//! An entity is encoded as:
//! - 1 byte format version
//! - 8 bytes entity ID (big-endian u64)
//! - 4 bytes label count
//! - For each label: 4 bytes length + UTF-8 bytes
//! - 4 bytes property count
//! - For each property: 4 bytes key length + key bytes + encoded value

use std::collections::HashMap;

use crate::error::CoreError;
use crate::types::{Entity, EntityId, Label};

use super::traits::{Decoder, Encoder, FORMAT_VERSION};
use super::value::decode_value;

impl Encoder for Entity {
    fn encode(&self) -> Result<Vec<u8>, CoreError> {
        let mut buf = Vec::new();
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    fn encode_to(&self, buf: &mut Vec<u8>) -> Result<(), CoreError> {
        // Format version
        buf.push(FORMAT_VERSION);

        // Entity ID
        buf.extend_from_slice(&self.id.as_u64().to_be_bytes());

        // Labels
        let label_count = u32::try_from(self.labels.len())
            .map_err(|_| CoreError::Encoding("too many labels".to_owned()))?;
        buf.extend_from_slice(&label_count.to_be_bytes());

        for label in &self.labels {
            let bytes = label.as_str().as_bytes();
            let len = u32::try_from(bytes.len())
                .map_err(|_| CoreError::Encoding("label too long".to_owned()))?;
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(bytes);
        }

        // Properties
        let prop_count = u32::try_from(self.properties.len())
            .map_err(|_| CoreError::Encoding("too many properties".to_owned()))?;
        buf.extend_from_slice(&prop_count.to_be_bytes());

        for (key, value) in &self.properties {
            let key_bytes = key.as_bytes();
            let key_len = u32::try_from(key_bytes.len())
                .map_err(|_| CoreError::Encoding("property key too long".to_owned()))?;
            buf.extend_from_slice(&key_len.to_be_bytes());
            buf.extend_from_slice(key_bytes);
            value.encode_to(buf)?;
        }

        Ok(())
    }
}

impl Decoder for Entity {
    fn decode(bytes: &[u8]) -> Result<Self, CoreError> {
        if bytes.is_empty() {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }

        let version = bytes[0];
        if version != FORMAT_VERSION {
            return Err(CoreError::Encoding(format!(
                "unsupported format version: {version}, expected {FORMAT_VERSION}"
            )));
        }

        let mut offset = 1;

        // Entity ID
        if bytes.len() < offset + 8 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let id_bytes: [u8; 8] = bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read entity ID".to_owned()))?;
        let id = EntityId::new(u64::from_be_bytes(id_bytes));
        offset += 8;

        // Labels
        if bytes.len() < offset + 4 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let label_count_bytes: [u8; 4] = bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read label count".to_owned()))?;
        let label_count = u32::from_be_bytes(label_count_bytes) as usize;
        offset += 4;

        let mut labels = Vec::with_capacity(label_count);
        for _ in 0..label_count {
            if bytes.len() < offset + 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let len_bytes: [u8; 4] = bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read label length".to_owned()))?;
            let len = u32::from_be_bytes(len_bytes) as usize;
            offset += 4;

            if bytes.len() < offset + len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let label_str = String::from_utf8(bytes[offset..offset + len].to_vec())
                .map_err(|e| CoreError::Encoding(format!("invalid label UTF-8: {e}")))?;
            labels.push(Label::new(label_str));
            offset += len;
        }

        // Properties
        if bytes.len() < offset + 4 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let prop_count_bytes: [u8; 4] = bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read property count".to_owned()))?;
        let prop_count = u32::from_be_bytes(prop_count_bytes) as usize;
        offset += 4;

        let mut properties = HashMap::with_capacity(prop_count);
        for _ in 0..prop_count {
            if bytes.len() < offset + 4 {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let key_len_bytes: [u8; 4] = bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| CoreError::Encoding("failed to read key length".to_owned()))?;
            let key_len = u32::from_be_bytes(key_len_bytes) as usize;
            offset += 4;

            if bytes.len() < offset + key_len {
                return Err(CoreError::Encoding("unexpected end of input".to_owned()));
            }
            let key = String::from_utf8(bytes[offset..offset + key_len].to_vec())
                .map_err(|e| CoreError::Encoding(format!("invalid key UTF-8: {e}")))?;
            offset += key_len;

            let (value, consumed) = decode_value(&bytes[offset..])?;
            properties.insert(key, value);
            offset += consumed;
        }

        Ok(Self { id, labels, properties, vectors: HashMap::new() })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::types::Value;

    #[test]
    fn encode_decode_empty_entity() {
        let original = Entity::new(EntityId::new(42));
        let encoded = original.encode().unwrap();
        let decoded = Entity::decode(&encoded).unwrap();
        assert_eq!(decoded.id, original.id);
        assert!(decoded.labels.is_empty());
        assert!(decoded.properties.is_empty());
    }

    #[test]
    fn encode_decode_entity_with_labels() {
        let original = Entity::new(EntityId::new(1)).with_label("Person").with_label("Employee");
        let encoded = original.encode().unwrap();
        let decoded = Entity::decode(&encoded).unwrap();
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.labels.len(), 2);
        assert!(decoded.has_label("Person"));
        assert!(decoded.has_label("Employee"));
    }

    #[test]
    fn encode_decode_entity_with_properties() {
        let original = Entity::new(EntityId::new(1))
            .with_property("name", "Alice")
            .with_property("age", 30i64)
            .with_property("active", true);
        let encoded = original.encode().unwrap();
        let decoded = Entity::decode(&encoded).unwrap();
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.get_property("name"), Some(&Value::String("Alice".to_owned())));
        assert_eq!(decoded.get_property("age"), Some(&Value::Int(30)));
        assert_eq!(decoded.get_property("active"), Some(&Value::Bool(true)));
    }

    #[test]
    fn encode_decode_full_entity() {
        let original = Entity::new(EntityId::new(12345))
            .with_label("Person")
            .with_property("name", "Bob")
            .with_property("embedding", Value::Vector(vec![0.1, 0.2, 0.3]));
        let encoded = original.encode().unwrap();
        let decoded = Entity::decode(&encoded).unwrap();
        assert_eq!(decoded.id, original.id);
        assert!(decoded.has_label("Person"));
        assert_eq!(decoded.get_property("name"), Some(&Value::String("Bob".to_owned())));
        assert_eq!(decoded.get_property("embedding"), Some(&Value::Vector(vec![0.1, 0.2, 0.3])));
    }

    #[test]
    fn decode_wrong_version() {
        let mut encoded = Entity::new(EntityId::new(1)).encode().unwrap();
        encoded[0] = 99; // Invalid version
        let result = Entity::decode(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn decode_empty_input() {
        let result = Entity::decode(&[]);
        assert!(result.is_err());
    }
}
