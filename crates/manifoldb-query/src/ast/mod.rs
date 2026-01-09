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
    AggregateWindowFunction, BinaryOp, CaseExpr, Expr, FunctionCall, HybridCombinationMethod,
    HybridSearchComponent, Identifier, Literal, MapProjectionItem, OrderByExpr, ParameterRef,
    QualifiedName, UnaryOp, WindowFrame, WindowFrameBound, WindowFrameUnits, WindowFunction,
    WindowSpec,
};
pub use pattern::{
    EdgeDirection, EdgeLength, EdgePattern, GraphPattern, NodePattern, PathPattern,
    PropertyCondition, ShortestPathPattern, WeightSpec,
};
pub use statement::{
    AlterColumnAction, AlterTableAction, AlterTableStatement, Assignment, CallStatement,
    ColumnConstraint, ColumnDef, ConflictAction, ConflictTarget, CreateCollectionStatement,
    CreateGraphStatement, CreateIndexStatement, CreateNodeRef, CreatePathStep, CreatePattern,
    CreateTableStatement, CreateViewStatement, DataType, DeleteGraphStatement, DeleteStatement,
    DropCollectionStatement, DropIndexStatement, DropTableStatement, DropViewStatement,
    ForeachAction, ForeachStatement, IndexColumn, InsertSource, InsertStatement, JoinClause,
    JoinCondition, JoinType, MatchStatement, MergeGraphStatement, MergePattern, OnConflict,
    PayloadFieldDef, RemoveGraphStatement, RemoveItem, ReturnItem, SelectItem, SelectStatement,
    SetAction, SetGraphStatement, SetOperation, SetOperator, Statement, TableAlias,
    TableConstraint, TableRef, UpdateStatement, VectorDef, VectorTypeDef, WithClause, YieldItem,
};
pub use vector::{
    DistanceMetric, VectorAggregate, VectorAggregateOp, VectorSearch, VectorSearchParams,
};
