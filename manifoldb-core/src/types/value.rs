//! Property values that can be stored on entities and edges.

use serde::{Deserialize, Serialize};

/// A value that can be stored as a property on an entity or edge.
///
/// This enum represents all possible value types in `ManifoldDB`, including
/// vector embeddings for similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Null/missing value
    Null,
    /// Boolean value
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point number
    Float(f64),
    /// UTF-8 string
    String(String),
    /// Raw bytes
    Bytes(Vec<u8>),
    /// Vector embedding (for similarity search)
    Vector(Vec<f32>),
    /// Array of values
    Array(Vec<Value>),
}

impl Value {
    /// Returns `true` if the value is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns the value as a boolean if it is one.
    #[inline]
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the value as an integer if it is one.
    #[inline]
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the value as a float if it is one.
    #[inline]
    #[must_use]
    pub const fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the value as a string slice if it is one.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the value as a vector slice if it is one.
    #[inline]
    #[must_use]
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Self::Vector(v) => Some(v),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    #[inline]
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i64> for Value {
    #[inline]
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<f64> for Value {
    #[inline]
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<String> for Value {
    #[inline]
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for Value {
    #[inline]
    fn from(s: &str) -> Self {
        Self::String(s.to_owned())
    }
}

impl From<Vec<f32>> for Value {
    #[inline]
    fn from(v: Vec<f32>) -> Self {
        Self::Vector(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_type_checks() {
        assert!(Value::Null.is_null());
        assert!(!Value::Bool(true).is_null());
    }

    #[test]
    fn value_conversions() {
        assert_eq!(Value::from(true).as_bool(), Some(true));
        assert_eq!(Value::from(42i64).as_int(), Some(42));
        assert_eq!(Value::from(2.5f64).as_float(), Some(2.5));
        assert_eq!(Value::from("hello").as_str(), Some("hello"));
    }

    #[test]
    fn vector_value() {
        let embedding = vec![0.1, 0.2, 0.3];
        let value = Value::from(embedding.clone());
        assert_eq!(value.as_vector(), Some(embedding.as_slice()));
    }
}
