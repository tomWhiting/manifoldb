//! Serialization for [`Edge`] types.
//!
//! This module provides compact binary encoding for edges (relationships) in the graph.
//!
//! # Format
//!
//! An edge is encoded as:
//! - 1 byte format version
//! - 8 bytes edge ID (big-endian u64)
//! - 8 bytes source entity ID (big-endian u64)
//! - 8 bytes target entity ID (big-endian u64)
//! - 4 bytes edge type length + UTF-8 bytes
//! - 4 bytes property count
//! - For each property: 4 bytes key length + key bytes + encoded value

use std::collections::HashMap;

use crate::error::CoreError;
use crate::types::{Edge, EdgeId, EdgeType, EntityId};

use super::traits::{Decoder, Encoder, FORMAT_VERSION};
use super::value::decode_value;

impl Encoder for Edge {
    fn encode(&self) -> Result<Vec<u8>, CoreError> {
        let mut buf = Vec::new();
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    fn encode_to(&self, buf: &mut Vec<u8>) -> Result<(), CoreError> {
        // Format version
        buf.push(FORMAT_VERSION);

        // Edge ID
        buf.extend_from_slice(&self.id.as_u64().to_be_bytes());

        // Source and target entity IDs
        buf.extend_from_slice(&self.source.as_u64().to_be_bytes());
        buf.extend_from_slice(&self.target.as_u64().to_be_bytes());

        // Edge type
        let type_bytes = self.edge_type.as_str().as_bytes();
        let type_len = u32::try_from(type_bytes.len())
            .map_err(|_| CoreError::Encoding("edge type too long".to_owned()))?;
        buf.extend_from_slice(&type_len.to_be_bytes());
        buf.extend_from_slice(type_bytes);

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

impl Decoder for Edge {
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

        // Edge ID
        if bytes.len() < offset + 8 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let id_bytes: [u8; 8] = bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read edge ID".to_owned()))?;
        let id = EdgeId::new(u64::from_be_bytes(id_bytes));
        offset += 8;

        // Source entity ID
        if bytes.len() < offset + 8 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let source_bytes: [u8; 8] = bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read source ID".to_owned()))?;
        let source = EntityId::new(u64::from_be_bytes(source_bytes));
        offset += 8;

        // Target entity ID
        if bytes.len() < offset + 8 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let target_bytes: [u8; 8] = bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read target ID".to_owned()))?;
        let target = EntityId::new(u64::from_be_bytes(target_bytes));
        offset += 8;

        // Edge type
        if bytes.len() < offset + 4 {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let type_len_bytes: [u8; 4] = bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| CoreError::Encoding("failed to read edge type length".to_owned()))?;
        let type_len = u32::from_be_bytes(type_len_bytes) as usize;
        offset += 4;

        if bytes.len() < offset + type_len {
            return Err(CoreError::Encoding("unexpected end of input".to_owned()));
        }
        let edge_type_str = String::from_utf8(bytes[offset..offset + type_len].to_vec())
            .map_err(|e| CoreError::Encoding(format!("invalid edge type UTF-8: {e}")))?;
        let edge_type = EdgeType::new(edge_type_str);
        offset += type_len;

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

        Ok(Self {
            id,
            source,
            target,
            edge_type,
            properties,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::types::Value;

    #[test]
    fn encode_decode_simple_edge() {
        let original = Edge::new(EdgeId::new(1), EntityId::new(10), EntityId::new(20), "FOLLOWS");
        let encoded = original.encode().unwrap();
        let decoded = Edge::decode(&encoded).unwrap();
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.source, original.source);
        assert_eq!(decoded.target, original.target);
        assert_eq!(decoded.edge_type.as_str(), "FOLLOWS");
        assert!(decoded.properties.is_empty());
    }

    #[test]
    fn encode_decode_edge_with_properties() {
        let original =
            Edge::new(EdgeId::new(100), EntityId::new(1), EntityId::new(2), "KNOWS")
                .with_property("since", "2024-01-01")
                .with_property("weight", 0.8f64);
        let encoded = original.encode().unwrap();
        let decoded = Edge::decode(&encoded).unwrap();
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.source, original.source);
        assert_eq!(decoded.target, original.target);
        assert_eq!(decoded.edge_type.as_str(), "KNOWS");
        assert_eq!(
            decoded.get_property("since"),
            Some(&Value::String("2024-01-01".to_owned()))
        );
        assert_eq!(decoded.get_property("weight"), Some(&Value::Float(0.8)));
    }

    #[test]
    fn encode_decode_edge_with_max_ids() {
        let original = Edge::new(
            EdgeId::new(u64::MAX),
            EntityId::new(u64::MAX),
            EntityId::new(u64::MAX),
            "MAX",
        );
        let encoded = original.encode().unwrap();
        let decoded = Edge::decode(&encoded).unwrap();
        assert_eq!(decoded.id.as_u64(), u64::MAX);
        assert_eq!(decoded.source.as_u64(), u64::MAX);
        assert_eq!(decoded.target.as_u64(), u64::MAX);
    }

    #[test]
    fn decode_wrong_version() {
        let mut encoded =
            Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST")
                .encode()
                .unwrap();
        encoded[0] = 99; // Invalid version
        let result = Edge::decode(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn decode_empty_input() {
        let result = Edge::decode(&[]);
        assert!(result.is_err());
    }
}
