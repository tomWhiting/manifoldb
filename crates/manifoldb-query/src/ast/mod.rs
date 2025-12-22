//! Query abstract syntax tree.
//!
//! This module defines the AST types for parsed queries, including:
//!
//! - [`statement`] - Top-level statements (SELECT, INSERT, UPDATE, DELETE, etc.)
//! - [`expr`] - Expressions (literals, operators, function calls, etc.)
//! - [`pattern`] - Graph patterns for MATCH clauses
//! - [`vector`] - Vector similarity search operations

pub mod expr;
pub mod pattern;
pub mod statement;
pub mod vector;

// Re-export commonly used types at the module level
pub use expr::{
    BinaryOp, CaseExpr, Expr, FunctionCall, HybridCombinationMethod, HybridSearchComponent,
    Identifier, Literal, OrderByExpr, ParameterRef, QualifiedName, UnaryOp, WindowFrame,
    WindowFrameBound, WindowFrameUnits, WindowSpec,
};
pub use pattern::{
    EdgeDirection, EdgeLength, EdgePattern, GraphPattern, NodePattern, PathPattern,
    PropertyCondition, ShortestPathPattern, WeightSpec,
};
pub use statement::{
    Assignment, ColumnConstraint, ColumnDef, ConflictAction, ConflictTarget,
    CreateCollectionStatement, CreateIndexStatement, CreateTableStatement, DataType,
    DeleteStatement, DropCollectionStatement, DropIndexStatement, DropTableStatement, IndexColumn,
    InsertSource, InsertStatement, JoinClause, JoinCondition, JoinType, MatchStatement, OnConflict,
    ReturnItem, SelectItem, SelectStatement, SetOperation, SetOperator, Statement, TableAlias,
    TableConstraint, TableRef, UpdateStatement, VectorDef, VectorTypeDef,
};
pub use vector::{
    DistanceMetric, VectorAggregate, VectorAggregateOp, VectorSearch, VectorSearchParams,
};
