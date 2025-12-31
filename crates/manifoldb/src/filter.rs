//! Filter expressions for vector and entity search.
//!
//! This module provides filter expressions that can be used to narrow down
//! search results based on entity properties or payload fields.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::Filter;
//!
//! // Simple equality filter
//! let filter = Filter::eq("category", "programming");
//!
//! // Numeric range
//! let filter = Filter::range("price", Some(10.0), Some(100.0));
//!
//! // Combine with AND/OR
//! let filter = Filter::and([
//!     Filter::eq("category", "programming"),
//!     Filter::gte("rating", 4.0),
//! ]);
//! ```

use serde_json::Value as JsonValue;

/// A filter expression for narrowing search results.
///
/// Filters are applied to entity properties or payload fields and can be combined
/// using logical operators. Filters are evaluated against each entity's properties
/// during search.
///
/// # Example
///
/// ```ignore
/// use manifoldb::Filter;
///
/// // Find Rust symbols with high visibility
/// let filter = Filter::and([
///     Filter::eq("language", "rust"),
///     Filter::eq("visibility", "public"),
/// ]);
///
/// // Search with the filter
/// let results = db.search("dense")
///     .query(query_vector)
///     .filter(filter)
///     .limit(10)
///     .execute()?;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Match entities where field equals value.
    Eq {
        /// The field path in the properties.
        field: String,
        /// The value to match.
        value: JsonValue,
    },

    /// Match entities where field does not equal value.
    Ne {
        /// The field path in the properties.
        field: String,
        /// The value that should not match.
        value: JsonValue,
    },

    /// Match entities where field is greater than value.
    Gt {
        /// The field path in the properties.
        field: String,
        /// The threshold value (exclusive).
        value: f64,
    },

    /// Match entities where field is greater than or equal to value.
    Gte {
        /// The field path in the properties.
        field: String,
        /// The threshold value (inclusive).
        value: f64,
    },

    /// Match entities where field is less than value.
    Lt {
        /// The field path in the properties.
        field: String,
        /// The threshold value (exclusive).
        value: f64,
    },

    /// Match entities where field is less than or equal to value.
    Lte {
        /// The field path in the properties.
        field: String,
        /// The threshold value (inclusive).
        value: f64,
    },

    /// Match entities where field is within a range.
    Range {
        /// The field path in the properties.
        field: String,
        /// The minimum value (inclusive).
        min: Option<f64>,
        /// The maximum value (inclusive).
        max: Option<f64>,
    },

    /// Match entities where field value is in the given set.
    In {
        /// The field path in the properties.
        field: String,
        /// The set of values to match.
        values: Vec<JsonValue>,
    },

    /// Match entities where field value is not in the given set.
    NotIn {
        /// The field path in the properties.
        field: String,
        /// The set of values to exclude.
        values: Vec<JsonValue>,
    },

    /// Match entities where string field contains substring.
    Contains {
        /// The field path in the properties.
        field: String,
        /// The substring to search for.
        substring: String,
    },

    /// Match entities where string field starts with prefix.
    StartsWith {
        /// The field path in the properties.
        field: String,
        /// The prefix to match.
        prefix: String,
    },

    /// Match entities where array field contains value.
    ArrayContains {
        /// The field path in the properties.
        field: String,
        /// The value to find in the array.
        value: JsonValue,
    },

    /// Match entities where field exists (is not null).
    Exists {
        /// The field path in the properties.
        field: String,
    },

    /// Match entities where field does not exist (is null).
    NotExists {
        /// The field path in the properties.
        field: String,
    },

    /// Match entities where all conditions are true.
    And(Vec<Filter>),

    /// Match entities where any condition is true.
    Or(Vec<Filter>),

    /// Match entities where condition is false.
    Not(Box<Filter>),
}

impl Filter {
    /// Create an equality filter.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Filter;
    ///
    /// let filter = Filter::eq("status", "active");
    /// ```
    pub fn eq(field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        Self::Eq { field: field.into(), value: value.into() }
    }

    /// Create a not-equal filter.
    pub fn ne(field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        Self::Ne { field: field.into(), value: value.into() }
    }

    /// Create a greater-than filter.
    pub fn gt(field: impl Into<String>, value: impl Into<f64>) -> Self {
        Self::Gt { field: field.into(), value: value.into() }
    }

    /// Create a greater-than-or-equal filter.
    pub fn gte(field: impl Into<String>, value: impl Into<f64>) -> Self {
        Self::Gte { field: field.into(), value: value.into() }
    }

    /// Create a less-than filter.
    pub fn lt(field: impl Into<String>, value: impl Into<f64>) -> Self {
        Self::Lt { field: field.into(), value: value.into() }
    }

    /// Create a less-than-or-equal filter.
    pub fn lte(field: impl Into<String>, value: impl Into<f64>) -> Self {
        Self::Lte { field: field.into(), value: value.into() }
    }

    /// Create a range filter.
    ///
    /// At least one of `min` or `max` should be specified.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Filter;
    ///
    /// // Price between 10 and 100
    /// let filter = Filter::range("price", Some(10.0), Some(100.0));
    ///
    /// // Age at least 18
    /// let filter = Filter::range("age", Some(18.0), None);
    /// ```
    pub fn range(field: impl Into<String>, min: Option<f64>, max: Option<f64>) -> Self {
        Self::Range { field: field.into(), min, max }
    }

    /// Create an "in" filter (field value in set).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Filter;
    ///
    /// let filter = Filter::in_set("category", ["fiction", "non-fiction"]);
    /// ```
    pub fn in_set<V: Into<JsonValue>>(
        field: impl Into<String>,
        values: impl IntoIterator<Item = V>,
    ) -> Self {
        Self::In { field: field.into(), values: values.into_iter().map(Into::into).collect() }
    }

    /// Create a "not in" filter (field value not in set).
    pub fn not_in<V: Into<JsonValue>>(
        field: impl Into<String>,
        values: impl IntoIterator<Item = V>,
    ) -> Self {
        Self::NotIn { field: field.into(), values: values.into_iter().map(Into::into).collect() }
    }

    /// Create a "contains" filter for string fields.
    pub fn contains(field: impl Into<String>, substring: impl Into<String>) -> Self {
        Self::Contains { field: field.into(), substring: substring.into() }
    }

    /// Create a "starts with" filter for string fields.
    pub fn starts_with(field: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self::StartsWith { field: field.into(), prefix: prefix.into() }
    }

    /// Create an "array contains" filter.
    pub fn array_contains(field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        Self::ArrayContains { field: field.into(), value: value.into() }
    }

    /// Create an "exists" filter (field is not null).
    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists { field: field.into() }
    }

    /// Create a "not exists" filter (field is null).
    pub fn not_exists(field: impl Into<String>) -> Self {
        Self::NotExists { field: field.into() }
    }

    /// Create an AND filter combining multiple conditions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Filter;
    ///
    /// let filter = Filter::and([
    ///     Filter::eq("category", "programming"),
    ///     Filter::gte("rating", 4.0),
    ///     Filter::lt("price", 50.0),
    /// ]);
    /// ```
    pub fn and(filters: impl IntoIterator<Item = Filter>) -> Self {
        Self::And(filters.into_iter().collect())
    }

    /// Create an OR filter combining multiple conditions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Filter;
    ///
    /// let filter = Filter::or([
    ///     Filter::eq("category", "fiction"),
    ///     Filter::eq("category", "poetry"),
    /// ]);
    /// ```
    pub fn or(filters: impl IntoIterator<Item = Filter>) -> Self {
        Self::Or(filters.into_iter().collect())
    }

    /// Create a NOT filter negating a condition.
    pub fn not(filter: Filter) -> Self {
        Self::Not(Box::new(filter))
    }

    /// Combine this filter with another using AND.
    #[must_use]
    pub fn and_then(self, other: Filter) -> Self {
        match self {
            Self::And(mut filters) => {
                filters.push(other);
                Self::And(filters)
            }
            _ => Self::And(vec![self, other]),
        }
    }

    /// Combine this filter with another using OR.
    #[must_use]
    pub fn or_else(self, other: Filter) -> Self {
        match self {
            Self::Or(mut filters) => {
                filters.push(other);
                Self::Or(filters)
            }
            _ => Self::Or(vec![self, other]),
        }
    }

    /// Evaluate this filter against a JSON payload.
    ///
    /// Returns `true` if the payload matches the filter conditions.
    #[must_use]
    pub fn matches(&self, payload: &JsonValue) -> bool {
        match self {
            Self::Eq { field, value } => {
                get_field(payload, field).map(|v| values_equal(v, value)).unwrap_or(false)
            }

            Self::Ne { field, value } => {
                get_field(payload, field).map(|v| !values_equal(v, value)).unwrap_or(true)
            }

            Self::Gt { field, value } => get_field(payload, field)
                .and_then(|v| v.as_f64())
                .map(|v| v > *value)
                .unwrap_or(false),

            Self::Gte { field, value } => get_field(payload, field)
                .and_then(|v| v.as_f64())
                .map(|v| v >= *value)
                .unwrap_or(false),

            Self::Lt { field, value } => get_field(payload, field)
                .and_then(|v| v.as_f64())
                .map(|v| v < *value)
                .unwrap_or(false),

            Self::Lte { field, value } => get_field(payload, field)
                .and_then(|v| v.as_f64())
                .map(|v| v <= *value)
                .unwrap_or(false),

            Self::Range { field, min, max } => get_field(payload, field)
                .and_then(|v| v.as_f64())
                .map(|v| {
                    let above_min = min.map_or(true, |m| v >= m);
                    let below_max = max.map_or(true, |m| v <= m);
                    above_min && below_max
                })
                .unwrap_or(false),

            Self::In { field, values } => get_field(payload, field)
                .map(|v| values.iter().any(|val| values_equal(v, val)))
                .unwrap_or(false),

            Self::NotIn { field, values } => get_field(payload, field)
                .map(|v| !values.iter().any(|val| values_equal(v, val)))
                .unwrap_or(true),

            Self::Contains { field, substring } => get_field(payload, field)
                .and_then(|v| v.as_str())
                .map(|s| s.contains(substring.as_str()))
                .unwrap_or(false),

            Self::StartsWith { field, prefix } => get_field(payload, field)
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with(prefix.as_str()))
                .unwrap_or(false),

            Self::ArrayContains { field, value } => get_field(payload, field)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().any(|item| values_equal(item, value)))
                .unwrap_or(false),

            Self::Exists { field } => {
                get_field(payload, field).map(|v| !v.is_null()).unwrap_or(false)
            }

            Self::NotExists { field } => {
                get_field(payload, field).map(|v| v.is_null()).unwrap_or(true)
            }

            Self::And(filters) => filters.iter().all(|f| f.matches(payload)),

            Self::Or(filters) => filters.iter().any(|f| f.matches(payload)),

            Self::Not(filter) => !filter.matches(payload),
        }
    }

    /// Evaluate this filter against entity properties.
    ///
    /// Converts entity properties to JSON and evaluates the filter.
    #[must_use]
    pub fn matches_entity(&self, entity: &manifoldb_core::Entity) -> bool {
        // Convert entity properties to JSON for filtering
        let payload = entity_properties_to_json(entity);
        self.matches(&payload)
    }
}

/// Convert entity properties to a JSON value for filtering.
fn entity_properties_to_json(entity: &manifoldb_core::Entity) -> JsonValue {
    let mut map = serde_json::Map::new();
    for (key, value) in &entity.properties {
        map.insert(key.clone(), value_to_json(value));
    }
    JsonValue::Object(map)
}

/// Convert a manifoldb Value to a JSON value.
fn value_to_json(value: &manifoldb_core::Value) -> JsonValue {
    match value {
        manifoldb_core::Value::Null => JsonValue::Null,
        manifoldb_core::Value::Bool(b) => JsonValue::Bool(*b),
        manifoldb_core::Value::Int(i) => JsonValue::Number((*i).into()),
        manifoldb_core::Value::Float(f) => {
            serde_json::Number::from_f64(*f).map_or(JsonValue::Null, JsonValue::Number)
        }
        manifoldb_core::Value::String(s) => JsonValue::String(s.clone()),
        manifoldb_core::Value::Bytes(b) => {
            // Encode bytes as base64 string
            use base64::Engine;
            JsonValue::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        manifoldb_core::Value::Array(items) => {
            JsonValue::Array(items.iter().map(value_to_json).collect())
        }
        manifoldb_core::Value::SparseVector(pairs) => {
            // Encode sparse vector as array of [index, value] pairs
            JsonValue::Array(
                pairs
                    .iter()
                    .map(|(idx, val)| {
                        JsonValue::Array(vec![
                            JsonValue::Number((*idx).into()),
                            serde_json::Number::from_f64(*val as f64)
                                .map_or(JsonValue::Null, JsonValue::Number),
                        ])
                    })
                    .collect(),
            )
        }
        manifoldb_core::Value::MultiVector(vecs) => {
            // Encode multi-vector as array of arrays
            JsonValue::Array(
                vecs.iter()
                    .map(|v| {
                        JsonValue::Array(
                            v.iter()
                                .map(|f| {
                                    serde_json::Number::from_f64(*f as f64)
                                        .map_or(JsonValue::Null, JsonValue::Number)
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            )
        }
        manifoldb_core::Value::Vector(v) => JsonValue::Array(
            v.iter()
                .map(|f| {
                    JsonValue::Number(
                        serde_json::Number::from_f64(*f as f64)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    )
                })
                .collect(),
        ),
    }
}

/// Get a field value from a JSON payload by path.
///
/// Supports dot notation for nested fields (e.g., "metadata.author").
fn get_field<'a>(payload: &'a JsonValue, field: &str) -> Option<&'a JsonValue> {
    let mut current = payload;
    for part in field.split('.') {
        current = current.get(part)?;
    }
    Some(current)
}

/// Compare two JSON values for equality.
fn values_equal(a: &JsonValue, b: &JsonValue) -> bool {
    match (a, b) {
        (JsonValue::Number(a), JsonValue::Number(b)) => {
            // Compare numbers with tolerance for floating point
            match (a.as_f64(), b.as_f64()) {
                (Some(a), Some(b)) => (a - b).abs() < f64::EPSILON,
                _ => a == b,
            }
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_eq_filter() {
        let filter = Filter::eq("category", "programming");
        let payload = json!({"category": "programming", "title": "Rust Book"});
        assert!(filter.matches(&payload));

        let payload = json!({"category": "fiction"});
        assert!(!filter.matches(&payload));
    }

    #[test]
    fn test_numeric_filters() {
        let payload = json!({"price": 25.0, "rating": 4.5});

        assert!(Filter::gt("price", 20.0).matches(&payload));
        assert!(!Filter::gt("price", 30.0).matches(&payload));

        assert!(Filter::gte("price", 25.0).matches(&payload));
        assert!(!Filter::gte("price", 26.0).matches(&payload));

        assert!(Filter::lt("price", 30.0).matches(&payload));
        assert!(!Filter::lt("price", 20.0).matches(&payload));

        assert!(Filter::lte("price", 25.0).matches(&payload));
        assert!(!Filter::lte("price", 24.0).matches(&payload));
    }

    #[test]
    fn test_range_filter() {
        let filter = Filter::range("price", Some(10.0), Some(50.0));

        assert!(filter.matches(&json!({"price": 25.0})));
        assert!(filter.matches(&json!({"price": 10.0})));
        assert!(filter.matches(&json!({"price": 50.0})));
        assert!(!filter.matches(&json!({"price": 5.0})));
        assert!(!filter.matches(&json!({"price": 100.0})));
    }

    #[test]
    fn test_in_filter() {
        let filter = Filter::in_set("category", ["fiction", "poetry"]);

        assert!(filter.matches(&json!({"category": "fiction"})));
        assert!(filter.matches(&json!({"category": "poetry"})));
        assert!(!filter.matches(&json!({"category": "programming"})));
    }

    #[test]
    fn test_contains_filter() {
        let filter = Filter::contains("title", "Rust");

        assert!(filter.matches(&json!({"title": "The Rust Book"})));
        assert!(!filter.matches(&json!({"title": "Python Guide"})));
    }

    #[test]
    fn test_and_filter() {
        let filter =
            Filter::and([Filter::eq("category", "programming"), Filter::gte("rating", 4.0)]);

        assert!(filter.matches(&json!({"category": "programming", "rating": 4.5})));
        assert!(!filter.matches(&json!({"category": "programming", "rating": 3.5})));
        assert!(!filter.matches(&json!({"category": "fiction", "rating": 4.5})));
    }

    #[test]
    fn test_or_filter() {
        let filter =
            Filter::or([Filter::eq("category", "fiction"), Filter::eq("category", "poetry")]);

        assert!(filter.matches(&json!({"category": "fiction"})));
        assert!(filter.matches(&json!({"category": "poetry"})));
        assert!(!filter.matches(&json!({"category": "programming"})));
    }

    #[test]
    fn test_not_filter() {
        let filter = Filter::not(Filter::eq("status", "deleted"));

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(!filter.matches(&json!({"status": "deleted"})));
    }

    #[test]
    fn test_nested_field() {
        let filter = Filter::eq("metadata.author", "John");
        let payload = json!({"metadata": {"author": "John", "year": 2024}});
        assert!(filter.matches(&payload));
    }

    #[test]
    fn test_array_contains() {
        let filter = Filter::array_contains("tags", "rust");

        assert!(filter.matches(&json!({"tags": ["rust", "programming"]})));
        assert!(!filter.matches(&json!({"tags": ["python", "programming"]})));
    }

    #[test]
    fn test_exists() {
        let filter = Filter::exists("optional_field");

        assert!(filter.matches(&json!({"optional_field": "value"})));
        assert!(!filter.matches(&json!({"optional_field": null})));
        assert!(!filter.matches(&json!({"other_field": "value"})));
    }

    #[test]
    fn test_filter_chaining() {
        let filter = Filter::eq("category", "programming")
            .and_then(Filter::gte("rating", 4.0))
            .and_then(Filter::lt("price", 50.0));

        assert!(filter.matches(&json!({
            "category": "programming",
            "rating": 4.5,
            "price": 30.0
        })));
    }

    #[test]
    fn test_matches_entity() {
        use manifoldb_core::{Entity, EntityId};

        let entity = Entity::new(EntityId::new(1))
            .with_label("Test")
            .with_property("language", "rust")
            .with_property("rating", 4.5f64);

        let filter = Filter::and([Filter::eq("language", "rust"), Filter::gte("rating", 4.0)]);

        assert!(filter.matches_entity(&entity));

        let filter2 = Filter::eq("language", "python");
        assert!(!filter2.matches_entity(&entity));
    }
}
