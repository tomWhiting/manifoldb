//! Type system for logical plan nodes.
//!
//! This module provides the schema and type infrastructure for the logical plan.
//! Each plan node can compute its output schema, enabling type checking and
//! validation throughout the query planning process.
//!
//! # Overview
//!
//! The type system includes:
//! - [`PlanType`]: The data types supported in the query engine
//! - [`TypedColumn`]: A column with name and type information
//! - [`Schema`]: A collection of typed columns representing a relation's structure
//! - Type inference for expressions via [`LogicalExpr::infer_type`]
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_query::plan::logical::{LogicalPlan, Schema, TypedColumn, PlanType};
//!
//! // Create a schema for a users table
//! let schema = Schema::new(vec![
//!     TypedColumn::new("id", PlanType::BigInt),
//!     TypedColumn::new("name", PlanType::Text),
//!     TypedColumn::new("age", PlanType::Integer),
//! ]);
//!
//! // Check if a column exists
//! assert!(schema.field("name").is_some());
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::ast::DataType;

/// Data types used in the query planner's type system.
///
/// This is separate from AST `DataType` to allow for plan-specific type
/// representations (like `Any` for unknown types during planning).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlanType {
    /// Boolean type.
    Boolean,
    /// 16-bit signed integer.
    SmallInt,
    /// 32-bit signed integer.
    Integer,
    /// 64-bit signed integer.
    BigInt,
    /// 32-bit floating point.
    Real,
    /// 64-bit floating point.
    DoublePrecision,
    /// Arbitrary precision numeric.
    Numeric {
        /// Total digits.
        precision: Option<u32>,
        /// Digits after decimal point.
        scale: Option<u32>,
    },
    /// Variable-length string with optional max length.
    Varchar(Option<u32>),
    /// Unlimited-length text.
    Text,
    /// Binary data.
    Bytea,
    /// Timestamp without timezone.
    Timestamp,
    /// Timestamp with timezone.
    TimestampTz,
    /// Date.
    Date,
    /// Time without timezone.
    Time,
    /// Time with timezone.
    TimeTz,
    /// Interval.
    Interval,
    /// JSON.
    Json,
    /// Binary JSON.
    Jsonb,
    /// UUID.
    Uuid,
    /// Dense vector with dimension.
    Vector(Option<u32>),
    /// Array of another type.
    Array(Box<PlanType>),
    /// Graph node reference.
    Node,
    /// Graph edge/relationship reference.
    Edge,
    /// Graph path (sequence of nodes and edges).
    Path,
    /// List type (Cypher-style, similar to Array but semantically different).
    List(Box<PlanType>),
    /// Map type (Cypher-style key-value mapping).
    Map {
        /// Key type (typically Text).
        key: Box<PlanType>,
        /// Value type.
        value: Box<PlanType>,
    },
    /// Null type (for NULL literals).
    Null,
    /// Any type (unknown, to be inferred later).
    Any,
    /// Custom/user-defined type.
    Custom(String),
}

impl PlanType {
    /// Returns true if this type is numeric.
    #[must_use]
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::SmallInt
                | Self::Integer
                | Self::BigInt
                | Self::Real
                | Self::DoublePrecision
                | Self::Numeric { .. }
        )
    }

    /// Returns true if this type is a string type.
    #[must_use]
    pub fn is_string(&self) -> bool {
        matches!(self, Self::Varchar(_) | Self::Text)
    }

    /// Returns true if this type is a temporal type.
    #[must_use]
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            Self::Timestamp
                | Self::TimestampTz
                | Self::Date
                | Self::Time
                | Self::TimeTz
                | Self::Interval
        )
    }

    /// Returns true if this type is a vector type.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_))
    }

    /// Returns true if this type is a graph type.
    #[must_use]
    pub fn is_graph(&self) -> bool {
        matches!(self, Self::Node | Self::Edge | Self::Path)
    }

    /// Returns true if this type is a collection type (array, list, or map).
    #[must_use]
    pub fn is_collection(&self) -> bool {
        matches!(self, Self::Array(_) | Self::List(_) | Self::Map { .. })
    }

    /// Returns the element type if this is an array or list type.
    #[must_use]
    pub fn element_type(&self) -> Option<&PlanType> {
        match self {
            Self::Array(inner) | Self::List(inner) => Some(inner),
            _ => None,
        }
    }

    /// Returns true if this type can be compared with another type.
    #[must_use]
    pub fn is_comparable_to(&self, other: &Self) -> bool {
        // Any can be compared to anything
        if matches!(self, Self::Any) || matches!(other, Self::Any) {
            return true;
        }
        // Null can be compared to anything
        if matches!(self, Self::Null) || matches!(other, Self::Null) {
            return true;
        }
        // Same types are always comparable
        if self == other {
            return true;
        }
        // Numeric types are comparable to each other
        if self.is_numeric() && other.is_numeric() {
            return true;
        }
        // String types are comparable to each other
        if self.is_string() && other.is_string() {
            return true;
        }
        // Temporal types within the same category
        if self.is_temporal() && other.is_temporal() {
            // Date and Timestamp are comparable
            let date_like_self = matches!(self, Self::Date | Self::Timestamp | Self::TimestampTz);
            let date_like_other = matches!(other, Self::Date | Self::Timestamp | Self::TimestampTz);
            if date_like_self && date_like_other {
                return true;
            }
            // Time types are comparable
            let time_like_self = matches!(self, Self::Time | Self::TimeTz);
            let time_like_other = matches!(other, Self::Time | Self::TimeTz);
            if time_like_self && time_like_other {
                return true;
            }
        }
        false
    }

    /// Determines the common type for a binary operation between two types.
    ///
    /// Returns the "wider" type that both can be coerced to, or None if
    /// the types are incompatible.
    #[must_use]
    pub fn common_type(&self, other: &Self) -> Option<Self> {
        if self == other {
            return Some(self.clone());
        }
        // Any type adapts to the other
        if matches!(self, Self::Any) {
            return Some(other.clone());
        }
        if matches!(other, Self::Any) {
            return Some(self.clone());
        }
        // Null type adapts to the other
        if matches!(self, Self::Null) {
            return Some(other.clone());
        }
        if matches!(other, Self::Null) {
            return Some(self.clone());
        }
        // Numeric type promotion
        if self.is_numeric() && other.is_numeric() {
            return Some(promote_numeric(self, other));
        }
        // String types unify to Text
        if self.is_string() && other.is_string() {
            return Some(Self::Text);
        }
        // Temporal type promotion
        if matches!(self, Self::Date) && matches!(other, Self::Timestamp | Self::TimestampTz) {
            return Some(other.clone());
        }
        if matches!(other, Self::Date) && matches!(self, Self::Timestamp | Self::TimestampTz) {
            return Some(self.clone());
        }
        if matches!(self, Self::Timestamp) && matches!(other, Self::TimestampTz) {
            return Some(Self::TimestampTz);
        }
        if matches!(other, Self::Timestamp) && matches!(self, Self::TimestampTz) {
            return Some(Self::TimestampTz);
        }

        None
    }
}

/// Promotes two numeric types to their common supertype.
#[allow(clippy::enum_glob_use)]
fn promote_numeric(a: &PlanType, b: &PlanType) -> PlanType {
    use PlanType::*;

    // Order of numeric types from smallest to largest
    fn rank(t: &PlanType) -> u8 {
        match t {
            SmallInt => 1,
            Integer => 2,
            BigInt => 3,
            Real => 4,
            DoublePrecision => 5,
            Numeric { .. } => 6,
            _ => 0,
        }
    }

    // Float types supersede integer types
    let a_is_float = matches!(a, Real | DoublePrecision);
    let b_is_float = matches!(b, Real | DoublePrecision);
    let a_is_numeric = matches!(a, Numeric { .. });
    let b_is_numeric = matches!(b, Numeric { .. });

    // Numeric (decimal) takes precedence when mixed with floats
    if a_is_numeric || b_is_numeric {
        // Return the Numeric type with appropriate precision
        return if a_is_numeric { a.clone() } else { b.clone() };
    }

    // Float promotion
    if a_is_float || b_is_float {
        if matches!(a, DoublePrecision) || matches!(b, DoublePrecision) {
            return DoublePrecision;
        }
        return Real;
    }

    // Integer promotion
    if rank(a) >= rank(b) {
        a.clone()
    } else {
        b.clone()
    }
}

impl From<&DataType> for PlanType {
    fn from(dt: &DataType) -> Self {
        match dt {
            DataType::Boolean => Self::Boolean,
            DataType::SmallInt => Self::SmallInt,
            DataType::Integer => Self::Integer,
            DataType::BigInt => Self::BigInt,
            DataType::Real => Self::Real,
            DataType::DoublePrecision => Self::DoublePrecision,
            DataType::Numeric { precision, scale } => {
                Self::Numeric { precision: *precision, scale: *scale }
            }
            DataType::Varchar(len) => Self::Varchar(*len),
            DataType::Text => Self::Text,
            DataType::Bytea => Self::Bytea,
            DataType::Timestamp => Self::Timestamp,
            DataType::Date => Self::Date,
            DataType::Time => Self::Time,
            DataType::Interval => Self::Interval,
            DataType::Json => Self::Json,
            DataType::Jsonb => Self::Jsonb,
            DataType::Uuid => Self::Uuid,
            DataType::Vector(dim) => Self::Vector(*dim),
            DataType::Array(inner) => Self::Array(Box::new(Self::from(inner.as_ref()))),
            DataType::Custom(name) => Self::Custom(name.clone()),
        }
    }
}

impl From<DataType> for PlanType {
    fn from(dt: DataType) -> Self {
        Self::from(&dt)
    }
}

impl fmt::Display for PlanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::SmallInt => write!(f, "SMALLINT"),
            Self::Integer => write!(f, "INTEGER"),
            Self::BigInt => write!(f, "BIGINT"),
            Self::Real => write!(f, "REAL"),
            Self::DoublePrecision => write!(f, "DOUBLE PRECISION"),
            Self::Numeric { precision, scale } => {
                write!(f, "NUMERIC")?;
                if let Some(p) = precision {
                    write!(f, "({p}")?;
                    if let Some(s) = scale {
                        write!(f, ", {s}")?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Self::Varchar(len) => {
                write!(f, "VARCHAR")?;
                if let Some(l) = len {
                    write!(f, "({l})")?;
                }
                Ok(())
            }
            Self::Text => write!(f, "TEXT"),
            Self::Bytea => write!(f, "BYTEA"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::TimestampTz => write!(f, "TIMESTAMP WITH TIME ZONE"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::TimeTz => write!(f, "TIME WITH TIME ZONE"),
            Self::Interval => write!(f, "INTERVAL"),
            Self::Json => write!(f, "JSON"),
            Self::Jsonb => write!(f, "JSONB"),
            Self::Uuid => write!(f, "UUID"),
            Self::Vector(dim) => {
                write!(f, "VECTOR")?;
                if let Some(d) = dim {
                    write!(f, "({d})")?;
                }
                Ok(())
            }
            Self::Array(inner) => write!(f, "{inner}[]"),
            Self::Node => write!(f, "NODE"),
            Self::Edge => write!(f, "EDGE"),
            Self::Path => write!(f, "PATH"),
            Self::List(inner) => write!(f, "LIST<{inner}>"),
            Self::Map { key, value } => write!(f, "MAP<{key}, {value}>"),
            Self::Null => write!(f, "NULL"),
            Self::Any => write!(f, "ANY"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// A column in a schema with name and type information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedColumn {
    /// Column name.
    pub name: String,
    /// Optional qualifier (table alias or source name).
    pub qualifier: Option<String>,
    /// Column data type.
    pub data_type: PlanType,
    /// Whether the column can be NULL.
    pub nullable: bool,
}

impl TypedColumn {
    /// Creates a new typed column.
    #[must_use]
    pub fn new(name: impl Into<String>, data_type: PlanType) -> Self {
        Self { name: name.into(), qualifier: None, data_type, nullable: true }
    }

    /// Creates a new non-nullable typed column.
    #[must_use]
    pub fn new_non_null(name: impl Into<String>, data_type: PlanType) -> Self {
        Self { name: name.into(), qualifier: None, data_type, nullable: false }
    }

    /// Sets the qualifier for this column.
    #[must_use]
    pub fn with_qualifier(mut self, qualifier: impl Into<String>) -> Self {
        self.qualifier = Some(qualifier.into());
        self
    }

    /// Sets the nullability for this column.
    #[must_use]
    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Returns the qualified name of this column (e.g., "table.column").
    #[must_use]
    pub fn qualified_name(&self) -> String {
        match &self.qualifier {
            Some(q) => format!("{}.{}", q, self.name),
            None => self.name.clone(),
        }
    }
}

impl fmt::Display for TypedColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(q) = &self.qualifier {
            write!(f, "{}.{}: {}", q, self.name, self.data_type)?;
        } else {
            write!(f, "{}: {}", self.name, self.data_type)?;
        }
        if !self.nullable {
            write!(f, " NOT NULL")?;
        }
        Ok(())
    }
}

/// A schema representing the structure of a relation (table, query result, etc.).
///
/// The schema contains the columns with their names and types, and provides
/// methods for looking up columns by name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Schema {
    /// Columns in the schema.
    columns: Vec<TypedColumn>,
}

impl Schema {
    /// Creates a new schema with the given columns.
    #[must_use]
    pub fn new(columns: Vec<TypedColumn>) -> Self {
        Self { columns }
    }

    /// Creates an empty schema.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns the columns in this schema.
    #[must_use]
    pub fn columns(&self) -> &[TypedColumn] {
        &self.columns
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the schema has no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Looks up a column by name (unqualified).
    ///
    /// If multiple columns have the same name (from different qualifiers),
    /// returns the first match.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&TypedColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Looks up a column by qualified name (e.g., "table.column").
    #[must_use]
    pub fn field_qualified(&self, qualifier: &str, name: &str) -> Option<&TypedColumn> {
        self.columns.iter().find(|c| c.qualifier.as_deref() == Some(qualifier) && c.name == name)
    }

    /// Looks up a column by index.
    #[must_use]
    pub fn field_at(&self, index: usize) -> Option<&TypedColumn> {
        self.columns.get(index)
    }

    /// Returns the index of a column by name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Returns the index of a column by qualified name.
    #[must_use]
    pub fn index_of_qualified(&self, qualifier: &str, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.qualifier.as_deref() == Some(qualifier) && c.name == name)
    }

    /// Returns true if the schema contains a column with the given name.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.field(name).is_some()
    }

    /// Merges two schemas, combining all columns from both.
    ///
    /// This is used for JOIN operations where both input schemas contribute columns.
    #[must_use]
    pub fn merge(&self, other: &Schema) -> Schema {
        let mut columns = self.columns.clone();
        columns.extend(other.columns.iter().cloned());
        Schema { columns }
    }

    /// Creates a new schema with the columns requalified.
    #[must_use]
    pub fn with_qualifier(&self, qualifier: impl Into<String>) -> Schema {
        let q = qualifier.into();
        Schema {
            columns: self.columns.iter().map(|c| c.clone().with_qualifier(q.clone())).collect(),
        }
    }

    /// Creates a schema with selected columns by name.
    ///
    /// Returns None if any column name is not found.
    #[must_use]
    pub fn select(&self, column_names: &[&str]) -> Option<Schema> {
        let columns: Option<Vec<_>> =
            column_names.iter().map(|name| self.field(name).cloned()).collect();
        columns.map(|cols| Schema { columns: cols })
    }

    /// Creates a schema with columns projected by indices.
    #[must_use]
    pub fn project(&self, indices: &[usize]) -> Schema {
        let columns: Vec<_> =
            indices.iter().filter_map(|&i| self.columns.get(i).cloned()).collect();
        Schema { columns }
    }

    /// Returns an iterator over (name, type) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PlanType)> {
        self.columns.iter().map(|c| (c.name.as_str(), &c.data_type))
    }

    /// Returns a map of column names to types.
    #[must_use]
    pub fn to_type_map(&self) -> HashMap<String, PlanType> {
        self.columns.iter().map(|c| (c.name.clone(), c.data_type.clone())).collect()
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, col) in self.columns.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{col}")?;
        }
        write!(f, ")")
    }
}

impl IntoIterator for Schema {
    type Item = TypedColumn;
    type IntoIter = std::vec::IntoIter<TypedColumn>;

    fn into_iter(self) -> Self::IntoIter {
        self.columns.into_iter()
    }
}

impl<'a> IntoIterator for &'a Schema {
    type Item = &'a TypedColumn;
    type IntoIter = std::slice::Iter<'a, TypedColumn>;

    fn into_iter(self) -> Self::IntoIter {
        self.columns.iter()
    }
}

impl FromIterator<TypedColumn> for Schema {
    fn from_iter<T: IntoIterator<Item = TypedColumn>>(iter: T) -> Self {
        Schema { columns: iter.into_iter().collect() }
    }
}

/// Type checking context that holds schema information during type inference.
///
/// This is used when inferring expression types - it provides access to the
/// input schema(s) so column references can be resolved.
#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    /// Available schemas indexed by qualifier (alias).
    schemas: HashMap<String, Schema>,
    /// Default schema (used for unqualified column references).
    default_schema: Option<Schema>,
}

impl TypeContext {
    /// Creates a new empty type context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a type context with a single input schema.
    #[must_use]
    pub fn with_schema(schema: Schema) -> Self {
        Self { default_schema: Some(schema), schemas: HashMap::new() }
    }

    /// Adds a qualified schema to the context.
    pub fn add_schema(&mut self, qualifier: impl Into<String>, schema: Schema) {
        self.schemas.insert(qualifier.into(), schema);
    }

    /// Sets the default schema for unqualified column lookups.
    pub fn set_default_schema(&mut self, schema: Schema) {
        self.default_schema = Some(schema);
    }

    /// Looks up a column type by name.
    #[must_use]
    pub fn lookup_column(&self, qualifier: Option<&str>, name: &str) -> Option<&TypedColumn> {
        match qualifier {
            Some(q) => {
                // Look in the qualified schema
                self.schemas.get(q).and_then(|s| s.field(name))
            }
            None => {
                // First try the default schema
                if let Some(ref default) = self.default_schema {
                    if let Some(col) = default.field(name) {
                        return Some(col);
                    }
                }
                // Then search all schemas (might be ambiguous, but we return first match)
                for schema in self.schemas.values() {
                    if let Some(col) = schema.field(name) {
                        return Some(col);
                    }
                }
                None
            }
        }
    }

    /// Returns all available columns across all schemas.
    #[must_use]
    pub fn all_columns(&self) -> Vec<&TypedColumn> {
        let mut cols = Vec::new();
        if let Some(ref default) = self.default_schema {
            cols.extend(default.columns());
        }
        for schema in self.schemas.values() {
            cols.extend(schema.columns());
        }
        cols
    }

    /// Merges all schemas into a single combined schema.
    #[must_use]
    pub fn combined_schema(&self) -> Schema {
        let mut columns = Vec::new();
        if let Some(ref default) = self.default_schema {
            columns.extend(default.columns().iter().cloned());
        }
        for schema in self.schemas.values() {
            columns.extend(schema.columns().iter().cloned());
        }
        Schema::new(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_type_is_numeric() {
        assert!(PlanType::Integer.is_numeric());
        assert!(PlanType::BigInt.is_numeric());
        assert!(PlanType::Real.is_numeric());
        assert!(PlanType::DoublePrecision.is_numeric());
        assert!(!PlanType::Text.is_numeric());
        assert!(!PlanType::Boolean.is_numeric());
    }

    #[test]
    fn test_plan_type_is_comparable() {
        assert!(PlanType::Integer.is_comparable_to(&PlanType::BigInt));
        assert!(PlanType::Text.is_comparable_to(&PlanType::Varchar(Some(100))));
        assert!(PlanType::Date.is_comparable_to(&PlanType::Timestamp));
        assert!(!PlanType::Integer.is_comparable_to(&PlanType::Text));
        // Any is comparable to anything
        assert!(PlanType::Any.is_comparable_to(&PlanType::Integer));
        assert!(PlanType::Text.is_comparable_to(&PlanType::Any));
    }

    #[test]
    fn test_plan_type_common_type() {
        // Same types
        assert_eq!(PlanType::Integer.common_type(&PlanType::Integer), Some(PlanType::Integer));

        // Numeric promotion
        assert_eq!(PlanType::Integer.common_type(&PlanType::BigInt), Some(PlanType::BigInt));
        assert_eq!(
            PlanType::Integer.common_type(&PlanType::DoublePrecision),
            Some(PlanType::DoublePrecision)
        );

        // String unification
        assert_eq!(PlanType::Text.common_type(&PlanType::Varchar(Some(100))), Some(PlanType::Text));

        // Any adapts
        assert_eq!(PlanType::Any.common_type(&PlanType::Integer), Some(PlanType::Integer));
        assert_eq!(PlanType::Text.common_type(&PlanType::Any), Some(PlanType::Text));

        // Incompatible
        assert_eq!(PlanType::Integer.common_type(&PlanType::Text), None);
    }

    #[test]
    fn test_typed_column() {
        let col = TypedColumn::new("age", PlanType::Integer).with_qualifier("users");
        assert_eq!(col.name, "age");
        assert_eq!(col.qualifier, Some("users".to_string()));
        assert_eq!(col.qualified_name(), "users.age");
        assert!(col.nullable);

        let col2 = TypedColumn::new_non_null("id", PlanType::BigInt);
        assert!(!col2.nullable);
    }

    #[test]
    fn test_schema_basic() {
        let schema = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt),
            TypedColumn::new("name", PlanType::Text),
            TypedColumn::new("age", PlanType::Integer),
        ]);

        assert_eq!(schema.len(), 3);
        assert!(!schema.is_empty());
        assert!(schema.contains("id"));
        assert!(!schema.contains("email"));

        let col = schema.field("name").unwrap();
        assert_eq!(col.data_type, PlanType::Text);

        assert_eq!(schema.index_of("age"), Some(2));
    }

    #[test]
    fn test_schema_qualified_lookup() {
        let schema = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt).with_qualifier("users"),
            TypedColumn::new("id", PlanType::BigInt).with_qualifier("orders"),
            TypedColumn::new("name", PlanType::Text).with_qualifier("users"),
        ]);

        let col = schema.field_qualified("users", "id").unwrap();
        assert_eq!(col.qualifier, Some("users".to_string()));

        let col2 = schema.field_qualified("orders", "id").unwrap();
        assert_eq!(col2.qualifier, Some("orders".to_string()));

        assert!(schema.field_qualified("products", "id").is_none());
    }

    #[test]
    fn test_schema_merge() {
        let users = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt).with_qualifier("users"),
            TypedColumn::new("name", PlanType::Text).with_qualifier("users"),
        ]);
        let orders = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt).with_qualifier("orders"),
            TypedColumn::new("user_id", PlanType::BigInt).with_qualifier("orders"),
        ]);

        let merged = users.merge(&orders);
        assert_eq!(merged.len(), 4);
        assert!(merged.field_qualified("users", "id").is_some());
        assert!(merged.field_qualified("orders", "id").is_some());
    }

    #[test]
    fn test_schema_select() {
        let schema = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt),
            TypedColumn::new("name", PlanType::Text),
            TypedColumn::new("age", PlanType::Integer),
        ]);

        let selected = schema.select(&["id", "age"]).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.contains("id"));
        assert!(selected.contains("age"));
        assert!(!selected.contains("name"));

        // Non-existent column
        assert!(schema.select(&["id", "nonexistent"]).is_none());
    }

    #[test]
    fn test_type_context() {
        let users_schema = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt),
            TypedColumn::new("name", PlanType::Text),
        ]);
        let orders_schema = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt),
            TypedColumn::new("user_id", PlanType::BigInt),
        ]);

        let mut ctx = TypeContext::new();
        ctx.add_schema("users", users_schema);
        ctx.add_schema("orders", orders_schema);

        // Qualified lookup
        let col = ctx.lookup_column(Some("users"), "name").unwrap();
        assert_eq!(col.data_type, PlanType::Text);

        // Unqualified lookup (ambiguous id, returns first match)
        let col2 = ctx.lookup_column(None, "name").unwrap();
        assert_eq!(col2.data_type, PlanType::Text);

        // Not found
        assert!(ctx.lookup_column(None, "nonexistent").is_none());
    }

    #[test]
    fn test_plan_type_from_ast_datatype() {
        assert_eq!(PlanType::from(&DataType::Boolean), PlanType::Boolean);
        assert_eq!(PlanType::from(&DataType::Integer), PlanType::Integer);
        assert_eq!(PlanType::from(&DataType::Text), PlanType::Text);
        assert_eq!(PlanType::from(&DataType::Vector(Some(384))), PlanType::Vector(Some(384)));
        assert_eq!(
            PlanType::from(&DataType::Array(Box::new(DataType::Integer))),
            PlanType::Array(Box::new(PlanType::Integer))
        );
    }
}
