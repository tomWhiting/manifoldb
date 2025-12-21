//! Error types for query parsing and execution.

use thiserror::Error;

/// Errors that can occur during query parsing.
#[derive(Debug, Error)]
pub enum ParseError {
    /// An error from the underlying SQL parser.
    #[error("SQL syntax error: {0}")]
    SqlSyntax(String),

    /// Invalid graph pattern syntax.
    #[error("invalid graph pattern: {0}")]
    InvalidPattern(String),

    /// Invalid vector operation syntax.
    #[error("invalid vector operation: {0}")]
    InvalidVectorOp(String),

    /// Unsupported SQL feature.
    #[error("unsupported feature: {0}")]
    Unsupported(String),

    /// Empty query string.
    #[error("empty query")]
    EmptyQuery,

    /// Invalid identifier.
    #[error("invalid identifier: {0}")]
    InvalidIdentifier(String),

    /// Invalid literal value.
    #[error("invalid literal: {0}")]
    InvalidLiteral(String),

    /// Unexpected token during parsing.
    #[error("unexpected token: expected {expected}, found {found}")]
    UnexpectedToken {
        /// What was expected.
        expected: String,
        /// What was actually found.
        found: String,
    },

    /// Missing required clause.
    #[error("missing required clause: {0}")]
    MissingClause(String),

    /// Invalid operator for the given types.
    #[error("invalid operator {operator} for types {left_type} and {right_type}")]
    InvalidOperator {
        /// The operator that was used.
        operator: String,
        /// The left operand type.
        left_type: String,
        /// The right operand type.
        right_type: String,
    },
}

impl From<sqlparser::parser::ParserError> for ParseError {
    fn from(err: sqlparser::parser::ParserError) -> Self {
        Self::SqlSyntax(err.to_string())
    }
}

/// Result type for parsing operations.
pub type ParseResult<T> = Result<T, ParseError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = ParseError::SqlSyntax("unexpected EOF".to_string());
        assert!(err.to_string().contains("SQL syntax error"));
        assert!(err.to_string().contains("unexpected EOF"));
    }

    #[test]
    fn unexpected_token_display() {
        let err = ParseError::UnexpectedToken {
            expected: "identifier".to_string(),
            found: "number".to_string(),
        };
        assert!(err.to_string().contains("expected identifier"));
        assert!(err.to_string().contains("found number"));
    }
}
