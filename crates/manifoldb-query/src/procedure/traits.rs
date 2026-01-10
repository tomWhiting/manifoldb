//! Procedure trait and error types.

use std::sync::Arc;

use manifoldb_core::Value;
use thiserror::Error;

use super::signature::ProcedureSignature;
use crate::exec::{ExecutionContext, Row, RowBatch, Schema};

/// Errors that can occur during procedure execution.
#[derive(Debug, Error)]
pub enum ProcedureError {
    /// Procedure not found in registry.
    #[error("procedure not found: {0}")]
    NotFound(String),

    /// Invalid number of arguments.
    #[error("invalid argument count: expected {expected}, got {actual}")]
    InvalidArgCount {
        /// Expected number of arguments.
        expected: String,
        /// Actual number of arguments.
        actual: usize,
    },

    /// Invalid argument type.
    #[error("invalid argument type for parameter '{param}': expected {expected}, got {actual}")]
    InvalidArgType {
        /// The parameter name.
        param: String,
        /// Expected type.
        expected: String,
        /// Actual type.
        actual: String,
    },

    /// Invalid yield column.
    #[error("invalid yield column: '{column}' is not returned by procedure '{procedure}'")]
    InvalidYieldColumn {
        /// The requested column.
        column: String,
        /// The procedure name.
        procedure: String,
    },

    /// Procedure execution failed.
    #[error("procedure execution failed: {0}")]
    ExecutionFailed(String),

    /// Graph storage error.
    #[error("graph storage error: {0}")]
    GraphError(String),

    /// Internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type for procedure operations.
pub type ProcedureResult<T> = Result<T, ProcedureError>;

/// Arguments passed to a procedure.
#[derive(Debug, Clone)]
pub struct ProcedureArgs {
    /// The evaluated argument values.
    pub values: Vec<Value>,
}

impl ProcedureArgs {
    /// Creates new procedure arguments.
    #[must_use]
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }

    /// Returns the number of arguments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if there are no arguments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Gets an argument by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Gets an argument as a string, returning an error if not a string or missing.
    pub fn get_string(&self, index: usize, param_name: &str) -> ProcedureResult<&str> {
        match self.values.get(index) {
            Some(Value::String(s)) => Ok(s.as_str()),
            Some(other) => Err(ProcedureError::InvalidArgType {
                param: param_name.to_string(),
                expected: "STRING".to_string(),
                actual: value_type_name(other),
            }),
            None => Err(ProcedureError::InvalidArgCount {
                expected: format!("at least {}", index + 1),
                actual: self.values.len(),
            }),
        }
    }

    /// Gets an optional argument as a string.
    #[must_use]
    pub fn get_string_opt(&self, index: usize) -> Option<&str> {
        match self.values.get(index) {
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Gets an argument as an integer.
    pub fn get_int(&self, index: usize, param_name: &str) -> ProcedureResult<i64> {
        match self.values.get(index) {
            Some(Value::Int(i)) => Ok(*i),
            Some(other) => Err(ProcedureError::InvalidArgType {
                param: param_name.to_string(),
                expected: "INTEGER".to_string(),
                actual: value_type_name(other),
            }),
            None => Err(ProcedureError::InvalidArgCount {
                expected: format!("at least {}", index + 1),
                actual: self.values.len(),
            }),
        }
    }

    /// Gets an optional argument as an integer.
    #[must_use]
    pub fn get_int_opt(&self, index: usize) -> Option<i64> {
        match self.values.get(index) {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// Gets an argument as a float.
    pub fn get_float(&self, index: usize, param_name: &str) -> ProcedureResult<f64> {
        match self.values.get(index) {
            Some(Value::Float(f)) => Ok(*f),
            Some(Value::Int(i)) => Ok(*i as f64), // Allow integer promotion
            Some(other) => Err(ProcedureError::InvalidArgType {
                param: param_name.to_string(),
                expected: "FLOAT".to_string(),
                actual: value_type_name(other),
            }),
            None => Err(ProcedureError::InvalidArgCount {
                expected: format!("at least {}", index + 1),
                actual: self.values.len(),
            }),
        }
    }

    /// Gets an optional argument as a float with a default.
    #[must_use]
    pub fn get_float_or(&self, index: usize, default: f64) -> f64 {
        match self.values.get(index) {
            Some(Value::Float(f)) => *f,
            Some(Value::Int(i)) => *i as f64,
            _ => default,
        }
    }

    /// Gets an argument as an array.
    pub fn get_array(&self, index: usize, param_name: &str) -> ProcedureResult<&[Value]> {
        match self.values.get(index) {
            Some(Value::Array(arr)) => Ok(arr.as_slice()),
            Some(other) => Err(ProcedureError::InvalidArgType {
                param: param_name.to_string(),
                expected: "ARRAY".to_string(),
                actual: value_type_name(other),
            }),
            None => Err(ProcedureError::InvalidArgCount {
                expected: format!("at least {}", index + 1),
                actual: self.values.len(),
            }),
        }
    }

    /// Gets an optional argument as an array.
    #[must_use]
    pub fn get_array_opt(&self, index: usize) -> Option<&[Value]> {
        match self.values.get(index) {
            Some(Value::Array(arr)) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

/// Returns the type name of a value for error messages.
fn value_type_name(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Bool(_) => "BOOLEAN".to_string(),
        Value::Int(_) => "INTEGER".to_string(),
        Value::Float(_) => "FLOAT".to_string(),
        Value::String(_) => "STRING".to_string(),
        Value::Bytes(_) => "BYTES".to_string(),
        Value::Vector(_) => "VECTOR".to_string(),
        Value::SparseVector(_) => "SPARSE_VECTOR".to_string(),
        Value::MultiVector(_) => "MULTI_VECTOR".to_string(),
        Value::Array(_) => "ARRAY".to_string(),
        Value::Point { .. } => "POINT".to_string(),
        Value::Node { .. } => "NODE".to_string(),
        Value::Edge { .. } => "EDGE".to_string(),
    }
}

/// A callable procedure that can be invoked via CALL statements.
///
/// Procedures produce rows of results that can be consumed by YIELD clauses.
/// They are used for graph algorithms, introspection, and administrative tasks.
///
/// # Example Implementation
///
/// ```ignore
/// struct PageRankProcedure;
///
/// impl Procedure for PageRankProcedure {
///     fn signature(&self) -> ProcedureSignature {
///         ProcedureSignature::new("algo.pageRank")
///             .with_parameter(ProcedureParameter::required("node_label", "STRING"))
///             .with_parameter(ProcedureParameter::optional("damping", "FLOAT"))
///             .with_return(ReturnColumn::new("node", "NODE"))
///             .with_return(ReturnColumn::new("score", "FLOAT"))
///     }
///
///     fn execute(&self, args: ProcedureArgs) -> ProcedureResult<RowBatch> {
///         // Implementation...
///     }
/// }
/// ```
pub trait Procedure: Send + Sync {
    /// Returns the signature of this procedure.
    fn signature(&self) -> ProcedureSignature;

    /// Executes the procedure with the given arguments.
    ///
    /// Returns a batch of rows containing the procedure results.
    ///
    /// The default implementation calls `execute_with_context` with a new context.
    fn execute(&self, args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        self.execute_with_context(args, &ExecutionContext::new())
    }

    /// Executes the procedure with the given arguments and execution context.
    ///
    /// This method provides access to graph storage and other runtime context
    /// needed by graph algorithms and other procedures.
    ///
    /// The default implementation ignores the context and returns an error
    /// indicating the procedure needs to implement this method.
    fn execute_with_context(
        &self,
        _args: ProcedureArgs,
        _ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Default: return an error indicating the procedure needs to implement this method
        Err(ProcedureError::ExecutionFailed(format!(
            "procedure '{}' requires context but does not implement execute_with_context",
            self.signature().name
        )))
    }

    /// Returns the output schema for this procedure.
    ///
    /// By default, this is derived from the signature's return columns.
    fn output_schema(&self) -> Arc<Schema> {
        let columns: Vec<String> =
            self.signature().returns.iter().map(|c| c.name.clone()).collect();
        Arc::new(Schema::new(columns))
    }

    /// Returns whether this procedure requires execution context (e.g., graph access).
    ///
    /// Procedures that only need arguments can return `false` (default).
    /// Procedures that need graph storage or other context should return `true`.
    fn requires_context(&self) -> bool {
        false
    }
}

/// Helper to create a row from values matching a schema.
pub fn make_row(schema: Arc<Schema>, values: Vec<Value>) -> Row {
    Row::new(schema, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedure_args_get_string() {
        let args = ProcedureArgs::new(vec![Value::from("test"), Value::from(42i64)]);

        assert_eq!(args.get_string(0, "param").ok(), Some("test"));
        assert!(args.get_string(1, "param").is_err()); // Wrong type
        assert!(args.get_string(2, "param").is_err()); // Out of bounds
    }

    #[test]
    fn procedure_args_get_float() {
        let args = ProcedureArgs::new(vec![Value::from(3.14f64), Value::from(42i64)]);

        assert_eq!(args.get_float(0, "param").ok(), Some(3.14));
        assert_eq!(args.get_float(1, "param").ok(), Some(42.0)); // Int promotion
    }

    #[test]
    fn procedure_args_get_float_or() {
        let args = ProcedureArgs::new(vec![Value::from(0.5f64)]);

        assert_eq!(args.get_float_or(0, 0.85), 0.5);
        assert_eq!(args.get_float_or(1, 0.85), 0.85); // Default
    }
}
