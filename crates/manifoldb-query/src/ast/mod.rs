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
    EdgeDirection, EdgeLength, EdgePattern, GraphPattern, LabelExpression, NodePattern,
    PathPattern, PropertyCondition, ShortestPathPattern, WeightSpec,
};
pub use statement::{
    AlterColumnAction, AlterIndexAction, AlterIndexStatement, AlterSchemaAction,
    AlterSchemaStatement, AlterTableAction, AlterTableStatement, AnalyzeStatement, Assignment,
    BeginTransaction, CallStatement, ColumnConstraint, ColumnDef, ConflictAction, ConflictTarget,
    CopyDestination, CopyDirection, CopyFormat, CopyOptions, CopySource, CopyStatement, CopyTarget,
    CreateCollectionStatement, CreateFunctionStatement, CreateGraphStatement, CreateIndexStatement,
    CreateNodeRef, CreatePathStep, CreatePattern, CreateSchemaStatement, CreateTableStatement,
    CreateTriggerStatement, CreateViewStatement, DataType, DeleteGraphStatement, DeleteStatement,
    DropCollectionStatement, DropFunctionStatement, DropIndexStatement, DropSchemaStatement,
    DropTableStatement, DropTriggerStatement, DropViewStatement, ExplainAnalyzeStatement,
    ExplainFormat, ForeachAction, ForeachStatement, FunctionLanguage, FunctionParameter,
    FunctionVolatility, IndexColumn, InsertSource, InsertStatement, IsolationLevel, JoinClause,
    JoinCondition, JoinType, MatchStatement, MergeGraphStatement, MergePattern, OnConflict,
    ParameterMode, PartitionBound, PartitionBy, PartitionOf, PartitionRangeValue, PayloadFieldDef,
    ReleaseSavepointStatement, RemoveGraphStatement, RemoveItem, ResetStatement, ReturnItem,
    RollbackTransaction, SavepointStatement, SelectItem, SelectStatement, SetAction,
    SetGraphStatement, SetOperation, SetOperator, SetSearchPathStatement, SetSessionStatement,
    SetTransactionStatement, SetValue, ShowProceduresStatement, ShowStatement, Statement,
    TableAlias, TableConstraint, TableRef, TransactionAccessMode, TransactionStatement,
    TriggerEvent, TriggerForEach, TriggerTiming, TruncateCascade, TruncateIdentity,
    TruncateTableStatement, UpdateStatement, UtilityStatement, VacuumStatement, VectorDef,
    VectorTypeDef, WithClause, YieldItem,
};
pub use vector::{
    DistanceMetric, VectorAggregate, VectorAggregateOp, VectorSearch, VectorSearchParams,
};
