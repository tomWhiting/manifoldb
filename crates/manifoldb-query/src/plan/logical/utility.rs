//! Utility statement logical plan nodes.
//!
//! This module defines logical plan nodes for database utility operations:
//! - EXPLAIN ANALYZE (execution with statistics)
//! - VACUUM (table maintenance)
//! - ANALYZE (statistics collection)
//! - COPY (data import/export)
//! - SET/SHOW/RESET (session variables)

use super::node::LogicalPlan;
use super::validate::PlanResult;
use crate::ast::{
    CopyDestination, CopyDirection, CopyOptions, CopySource, CopyStatement, CopyTarget,
    QualifiedName, SetValue,
};

/// Output format for EXPLAIN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExplainFormat {
    /// Plain text format (default).
    #[default]
    Text,
    /// JSON format.
    Json,
    /// XML format.
    Xml,
    /// YAML format.
    Yaml,
}

impl std::fmt::Display for ExplainFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "TEXT"),
            Self::Json => write!(f, "JSON"),
            Self::Xml => write!(f, "XML"),
            Self::Yaml => write!(f, "YAML"),
        }
    }
}

/// EXPLAIN ANALYZE node.
///
/// Executes the inner plan and collects execution statistics.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExplainAnalyzeNode {
    /// The input plan to execute and analyze.
    pub input: Box<LogicalPlan>,
    /// Whether to include buffer usage statistics.
    pub buffers: bool,
    /// Whether to include timing information.
    pub timing: bool,
    /// Output format for the plan.
    pub format: ExplainFormat,
    /// Whether to show verbose output.
    pub verbose: bool,
    /// Whether to show cost estimates.
    pub costs: bool,
}

impl ExplainAnalyzeNode {
    /// Creates a new EXPLAIN ANALYZE node.
    #[must_use]
    pub fn new(input: LogicalPlan) -> Self {
        Self {
            input: Box::new(input),
            buffers: false,
            timing: true,
            format: ExplainFormat::Text,
            verbose: false,
            costs: true,
        }
    }
}

/// VACUUM node.
///
/// Reclaims storage and optionally updates table statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VacuumNode {
    /// Whether FULL vacuum is requested.
    pub full: bool,
    /// Whether to also collect statistics.
    pub analyze: bool,
    /// Target table (None means all tables).
    pub table: Option<QualifiedName>,
    /// Specific columns to analyze.
    pub columns: Vec<String>,
}

impl VacuumNode {
    /// Creates a VACUUM node for all tables.
    #[must_use]
    pub fn all() -> Self {
        Self { full: false, analyze: false, table: None, columns: vec![] }
    }

    /// Creates a VACUUM node for a specific table.
    #[must_use]
    pub fn table(name: QualifiedName) -> Self {
        Self { full: false, analyze: false, table: Some(name), columns: vec![] }
    }
}

/// ANALYZE node.
///
/// Collects statistics about table contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzeNode {
    /// Target table (None means all tables).
    pub table: Option<QualifiedName>,
    /// Specific columns to analyze.
    pub columns: Vec<String>,
}

impl AnalyzeNode {
    /// Creates an ANALYZE node for all tables.
    #[must_use]
    pub fn all() -> Self {
        Self { table: None, columns: vec![] }
    }

    /// Creates an ANALYZE node for a specific table.
    #[must_use]
    pub fn table(name: QualifiedName) -> Self {
        Self { table: Some(name), columns: vec![] }
    }
}

/// COPY node.
///
/// Handles data import/export operations.
#[derive(Debug, Clone, PartialEq)]
pub struct CopyNode {
    /// The copy target (table or query).
    pub target: CopyTarget,
    /// The copy direction.
    pub direction: CopyDirection,
    /// Copy options.
    pub options: CopyOptions,
    /// For COPY TO with a query, the query's logical plan.
    pub query_plan: Option<Box<LogicalPlan>>,
}

impl CopyNode {
    /// Creates a COPY node from an AST COPY statement.
    pub fn from_ast(stmt: &CopyStatement) -> PlanResult<Self> {
        Ok(Self {
            target: stmt.target.clone(),
            direction: stmt.direction.clone(),
            options: stmt.options.clone(),
            query_plan: None, // Query plan is built separately if needed
        })
    }

    /// Creates a COPY TO node for a table.
    #[must_use]
    pub fn table_to(table: QualifiedName, destination: String) -> Self {
        Self {
            target: CopyTarget::Table { name: table, columns: vec![] },
            direction: CopyDirection::To(CopyDestination::File(destination)),
            options: CopyOptions::default(),
            query_plan: None,
        }
    }

    /// Creates a COPY FROM node for a table.
    #[must_use]
    pub fn table_from(table: QualifiedName, source: String) -> Self {
        Self {
            target: CopyTarget::Table { name: table, columns: vec![] },
            direction: CopyDirection::From(CopySource::File(source)),
            options: CopyOptions::default(),
            query_plan: None,
        }
    }

    /// Whether this is a COPY TO (export) operation.
    #[must_use]
    pub fn is_export(&self) -> bool {
        matches!(self.direction, CopyDirection::To(_))
    }

    /// Whether this is a COPY FROM (import) operation.
    #[must_use]
    pub fn is_import(&self) -> bool {
        matches!(self.direction, CopyDirection::From(_))
    }
}

/// SET session variable node.
#[derive(Debug, Clone, PartialEq)]
pub struct SetSessionNode {
    /// The variable name.
    pub name: String,
    /// The value to set (None means DEFAULT).
    pub value: Option<SetValue>,
    /// Whether this is SET LOCAL (transaction-scoped).
    pub local: bool,
}

impl SetSessionNode {
    /// Creates a SET node with a string value.
    #[must_use]
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self { name: name.into(), value: Some(SetValue::string(value)), local: false }
    }

    /// Creates a SET node to DEFAULT.
    #[must_use]
    pub fn to_default(name: impl Into<String>) -> Self {
        Self { name: name.into(), value: None, local: false }
    }
}

/// SHOW session variable node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowNode {
    /// The variable to show (None means ALL).
    pub name: Option<String>,
}

impl ShowNode {
    /// Creates a SHOW ALL node.
    #[must_use]
    pub fn all() -> Self {
        Self { name: None }
    }

    /// Creates a SHOW node for a specific variable.
    #[must_use]
    pub fn variable(name: impl Into<String>) -> Self {
        Self { name: Some(name.into()) }
    }
}

/// RESET session variable node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResetNode {
    /// The variable to reset (None means ALL).
    pub name: Option<String>,
}

impl ResetNode {
    /// Creates a RESET ALL node.
    #[must_use]
    pub fn all() -> Self {
        Self { name: None }
    }

    /// Creates a RESET node for a specific variable.
    #[must_use]
    pub fn variable(name: impl Into<String>) -> Self {
        Self { name: Some(name.into()) }
    }
}

/// SHOW PROCEDURES node.
///
/// Lists available procedures in the database.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShowProceduresNode {
    /// Whether to only show executable procedures.
    pub executable: bool,
}

impl ShowProceduresNode {
    /// Creates a new SHOW PROCEDURES node.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the executable filter.
    #[must_use]
    pub const fn executable(mut self) -> Self {
        self.executable = true;
        self
    }
}
