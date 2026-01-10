//! SQL parser implementation.
//!
//! This module provides the core SQL parsing functionality using `sqlparser-rs`
//! as the foundation, with custom transformations to our AST types.

use sqlparser::ast as sp;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::ast::{
    AlterColumnAction, AlterIndexAction, AlterIndexStatement, AlterTableAction,
    AlterTableStatement, AnalyzeStatement, Assignment, BeginTransaction, BinaryOp, CallStatement,
    CaseExpr, ColumnConstraint, ColumnDef, ConflictAction, ConflictTarget, CopyDestination,
    CopyDirection, CopyFormat, CopyOptions, CopySource, CopyStatement, CopyTarget,
    CreateFunctionStatement, CreateIndexStatement, CreateSchemaStatement, CreateTableStatement,
    CreateTriggerStatement, CreateViewStatement, DataType, DeleteStatement, DropFunctionStatement,
    DropIndexStatement, DropSchemaStatement, DropTableStatement, DropTriggerStatement,
    DropViewStatement, ExplainAnalyzeStatement, ExplainFormat, Expr, FunctionCall,
    FunctionLanguage, FunctionParameter, FunctionVolatility, Identifier, IndexColumn, InsertSource,
    InsertStatement, IsolationLevel, JoinClause, JoinCondition, JoinType, Literal,
    NamedWindowDefinition, OnConflict, OrderByExpr, ParameterMode, ParameterRef, PartitionBy,
    PartitionOf, QualifiedName, ReleaseSavepointStatement, ResetStatement, RollbackTransaction,
    SavepointStatement, SelectItem, SelectStatement, SetOperation, SetOperator,
    SetSessionStatement, SetTransactionStatement, SetValue, ShowStatement, Statement, TableAlias,
    TableConstraint, TableRef, TransactionAccessMode, TransactionStatement, TriggerEvent,
    TriggerForEach, TriggerTiming, TruncateCascade, TruncateIdentity, TruncateTableStatement,
    UnaryOp, UpdateStatement, UtilityStatement, VacuumStatement, WindowFrame, WindowFrameBound,
    WindowFrameUnits, WindowSpec, WithClause,
};
use crate::error::{ParseError, ParseResult};

/// Parses a SQL string into a list of statements.
///
/// # Errors
///
/// Returns an error if the SQL is syntactically invalid.
pub fn parse_sql(sql: &str) -> ParseResult<Vec<Statement>> {
    if sql.trim().is_empty() {
        return Err(ParseError::EmptyQuery);
    }

    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)?;

    statements.into_iter().map(convert_statement).collect()
}

/// Parses a single SQL statement.
///
/// # Errors
///
/// Returns an error if the SQL is invalid or contains multiple statements.
pub fn parse_single_statement(sql: &str) -> ParseResult<Statement> {
    let mut stmts = parse_sql(sql)?;
    if stmts.len() != 1 {
        return Err(ParseError::SqlSyntax(format!("expected 1 statement, found {}", stmts.len())));
    }
    // SAFETY: We just verified there's exactly one statement
    Ok(stmts.remove(0))
}

/// Converts a sqlparser Statement to our Statement.
fn convert_statement(stmt: sp::Statement) -> ParseResult<Statement> {
    match stmt {
        sp::Statement::Query(query) => {
            let select = convert_query(*query)?;
            Ok(Statement::Select(Box::new(select)))
        }
        sp::Statement::Insert(insert) => {
            let insert_stmt = convert_insert(insert)?;
            Ok(Statement::Insert(Box::new(insert_stmt)))
        }
        sp::Statement::Update(update) => {
            let from_vec = match update.from {
                Some(sp::UpdateTableFromKind::BeforeSet(tables)) => Some(tables),
                Some(sp::UpdateTableFromKind::AfterSet(tables)) => Some(tables),
                None => None,
            };
            let update_stmt = convert_update(
                update.table,
                update.assignments,
                from_vec,
                update.selection,
                update.returning,
            )?;
            Ok(Statement::Update(Box::new(update_stmt)))
        }
        sp::Statement::Delete(delete) => {
            let delete_stmt = convert_delete(delete)?;
            Ok(Statement::Delete(Box::new(delete_stmt)))
        }
        sp::Statement::CreateTable(create) => {
            let create_stmt = convert_create_table(create)?;
            Ok(Statement::CreateTable(create_stmt))
        }
        sp::Statement::CreateIndex(create) => {
            let create_stmt = convert_create_index(create)?;
            Ok(Statement::CreateIndex(Box::new(create_stmt)))
        }
        sp::Statement::CreateView(create_view) => {
            if create_view.materialized {
                return Err(ParseError::Unsupported("MATERIALIZED VIEW".to_string()));
            }
            let view_stmt = convert_create_view(
                create_view.or_replace,
                create_view.name,
                create_view.columns,
                *create_view.query,
            )?;
            Ok(Statement::CreateView(Box::new(view_stmt)))
        }
        sp::Statement::AlterTable(sp::AlterTable { name, if_exists, operations, .. }) => {
            let alter_stmt = convert_alter_table(name, if_exists, operations)?;
            Ok(Statement::AlterTable(alter_stmt))
        }
        sp::Statement::AlterIndex { name, operation } => {
            let alter_stmt = convert_alter_index(name, operation)?;
            Ok(Statement::AlterIndex(alter_stmt))
        }
        sp::Statement::Truncate(truncate) => {
            let truncate_stmt = convert_truncate(truncate)?;
            Ok(Statement::TruncateTable(truncate_stmt))
        }
        sp::Statement::Drop { object_type, if_exists, names, cascade, .. } => match object_type {
            sp::ObjectType::Table => {
                let drop_stmt = DropTableStatement {
                    if_exists,
                    names: names.into_iter().map(convert_object_name).collect(),
                    cascade,
                };
                Ok(Statement::DropTable(drop_stmt))
            }
            sp::ObjectType::Index => {
                let drop_stmt = DropIndexStatement {
                    if_exists,
                    names: names.into_iter().map(convert_object_name).collect(),
                    cascade,
                };
                Ok(Statement::DropIndex(drop_stmt))
            }
            sp::ObjectType::View => {
                let drop_stmt = DropViewStatement {
                    if_exists,
                    names: names.into_iter().map(convert_object_name).collect(),
                    cascade,
                };
                Ok(Statement::DropView(drop_stmt))
            }
            sp::ObjectType::Schema => {
                // Schema names are simple identifiers, not qualified names
                let drop_stmt = DropSchemaStatement {
                    if_exists,
                    names: names.into_iter().map(|n| Identifier::new(n.to_string())).collect(),
                    cascade,
                };
                Ok(Statement::DropSchema(drop_stmt))
            }
            _ => Err(ParseError::Unsupported(format!("DROP {object_type:?}"))),
        },
        sp::Statement::Explain { statement, analyze, format, verbose, .. } => {
            let inner = convert_statement(*statement)?;
            // Check if ANALYZE option is present
            if analyze {
                let explain_stmt = convert_explain_analyze(inner, verbose, format)?;
                Ok(Statement::ExplainAnalyze(Box::new(explain_stmt)))
            } else {
                Ok(Statement::Explain(Box::new(inner)))
            }
        }
        sp::Statement::Call(function) => {
            let call_stmt = convert_call(function)?;
            Ok(Statement::Call(Box::new(call_stmt)))
        }
        // Utility statements
        sp::Statement::Copy { source, to, target, options, values, .. } => {
            let copy_stmt = convert_copy(source, to, target, options, values)?;
            Ok(Statement::Utility(Box::new(UtilityStatement::Copy(copy_stmt))))
        }
        sp::Statement::Set(sp::Set::SingleAssignment { variable, values, scope, .. }) => {
            let local = matches!(scope, Some(sp::ContextModifier::Local));
            let set_stmt = convert_set_single_assignment(variable, values, local)?;
            Ok(Statement::Utility(Box::new(UtilityStatement::Set(set_stmt))))
        }
        sp::Statement::Set(sp::Set::SetTransaction { modes, snapshot, session }) => {
            let txn_stmt = convert_set_transaction(modes, snapshot, session)?;
            Ok(Statement::Transaction(txn_stmt))
        }
        sp::Statement::Set(_) => Err(ParseError::Unsupported("SET statement variant".to_string())),
        sp::Statement::ShowVariable { variable } => {
            let show_stmt = convert_show_variable(variable)?;
            Ok(Statement::Utility(Box::new(UtilityStatement::Show(show_stmt))))
        }
        sp::Statement::Analyze(analyze) => {
            let analyze_stmt =
                convert_analyze(analyze.table_name, analyze.partitions, analyze.columns)?;
            Ok(Statement::Utility(Box::new(UtilityStatement::Analyze(analyze_stmt))))
        }
        sp::Statement::Vacuum(vacuum) => {
            let vacuum_stmt = convert_vacuum(vacuum)?;
            Ok(Statement::Utility(Box::new(UtilityStatement::Vacuum(vacuum_stmt))))
        }
        sp::Statement::Reset(reset) => {
            let reset_stmt = convert_reset(reset)?;
            Ok(Statement::Utility(Box::new(UtilityStatement::Reset(reset_stmt))))
        }
        // Transaction control statements
        sp::Statement::StartTransaction { modes, begin, .. } => {
            let txn_stmt = convert_start_transaction(modes, begin)?;
            Ok(Statement::Transaction(txn_stmt))
        }
        sp::Statement::Commit { .. } => Ok(Statement::Transaction(TransactionStatement::Commit)),
        sp::Statement::Rollback { chain: _, savepoint } => {
            let txn_stmt = convert_rollback(savepoint)?;
            Ok(Statement::Transaction(txn_stmt))
        }
        sp::Statement::Savepoint { name } => {
            let txn_stmt = TransactionStatement::Savepoint(SavepointStatement::new(
                Identifier::new(name.value),
            ));
            Ok(Statement::Transaction(txn_stmt))
        }
        sp::Statement::ReleaseSavepoint { name } => {
            let txn_stmt = TransactionStatement::ReleaseSavepoint(ReleaseSavepointStatement::new(
                Identifier::new(name.value),
            ));
            Ok(Statement::Transaction(txn_stmt))
        }
        // Schema DDL statements
        sp::Statement::CreateSchema { schema_name, if_not_exists, .. } => {
            let create_stmt = convert_create_schema(schema_name, if_not_exists)?;
            Ok(Statement::CreateSchema(create_stmt))
        }
        // Function DDL statements
        sp::Statement::CreateFunction(create_func) => {
            let create_stmt = convert_create_function(create_func)?;
            Ok(Statement::CreateFunction(Box::new(create_stmt)))
        }
        sp::Statement::DropFunction(drop_func) => {
            let drop_stmt = convert_drop_function(drop_func)?;
            Ok(Statement::DropFunction(drop_stmt))
        }
        // Trigger DDL statements
        sp::Statement::CreateTrigger(create_trigger) => {
            let create_stmt = convert_create_trigger(create_trigger)?;
            Ok(Statement::CreateTrigger(Box::new(create_stmt)))
        }
        sp::Statement::DropTrigger(drop_trigger) => {
            let drop_stmt = convert_drop_trigger(drop_trigger)?;
            Ok(Statement::DropTrigger(drop_stmt))
        }
        _ => Err(ParseError::Unsupported(format!("statement type: {stmt:?}"))),
    }
}

/// Converts a sqlparser Query to our `SelectStatement`.
fn convert_query(query: sp::Query) -> ParseResult<SelectStatement> {
    // Handle WITH clause if present
    let with_clauses =
        if let Some(with) = query.with { convert_with_clause(with)? } else { vec![] };

    let body = match *query.body {
        sp::SetExpr::Select(select) => convert_select(*select)?,
        sp::SetExpr::Query(subquery) => convert_query(*subquery)?,
        sp::SetExpr::SetOperation { op, set_quantifier, left, right } => {
            let mut base = match *left {
                sp::SetExpr::Select(select) => convert_select(*select)?,
                sp::SetExpr::Query(q) => convert_query(*q)?,
                _ => return Err(ParseError::Unsupported("nested set operation".to_string())),
            };
            let right_stmt = match *right {
                sp::SetExpr::Select(select) => convert_select(*select)?,
                sp::SetExpr::Query(q) => convert_query(*q)?,
                _ => return Err(ParseError::Unsupported("nested set operation".to_string())),
            };
            let set_op = SetOperation {
                op: match op {
                    sp::SetOperator::Union => SetOperator::Union,
                    sp::SetOperator::Intersect => SetOperator::Intersect,
                    sp::SetOperator::Except | sp::SetOperator::Minus => SetOperator::Except,
                },
                all: matches!(set_quantifier, sp::SetQuantifier::All),
                right: right_stmt,
            };
            base.set_op = Some(Box::new(set_op));
            base
        }
        sp::SetExpr::Values(values) => {
            // VALUES as a standalone select
            let rows: Vec<Vec<Expr>> = values
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>())
                .collect::<ParseResult<Vec<_>>>()?;

            if rows.is_empty() {
                return Err(ParseError::SqlSyntax("empty VALUES".to_string()));
            }

            // Create column aliases (column1, column2, etc.)
            let num_cols = rows.first().map_or(0, Vec::len);
            let projection: Vec<SelectItem> = (1..=num_cols)
                .map(|i| SelectItem::Expr {
                    expr: Expr::Column(QualifiedName::simple(format!("column{i}"))),
                    alias: None,
                })
                .collect();

            SelectStatement::new(projection)
        }
        _ => return Err(ParseError::Unsupported("set expression type".to_string())),
    };

    // Apply ORDER BY, LIMIT, OFFSET from the outer query
    let mut result = body;

    if let Some(order_by) = query.order_by {
        match order_by.kind {
            sp::OrderByKind::Expressions(exprs) => {
                result.order_by = exprs
                    .into_iter()
                    .map(convert_order_by_expr)
                    .collect::<ParseResult<Vec<_>>>()?;
            }
            sp::OrderByKind::All(_) => {
                return Err(ParseError::Unsupported("ORDER BY ALL".to_string()));
            }
        }
    }

    // Handle limit/offset via LimitClause
    if let Some(limit_clause) = query.limit_clause {
        match limit_clause {
            sp::LimitClause::LimitOffset { limit, offset, .. } => {
                if let Some(limit_expr) = limit {
                    result.limit = Some(convert_expr(limit_expr)?);
                }
                if let Some(offset_val) = offset {
                    result.offset = Some(convert_expr(offset_val.value)?);
                }
            }
            sp::LimitClause::OffsetCommaLimit { offset, limit } => {
                result.offset = Some(convert_expr(offset)?);
                result.limit = Some(convert_expr(limit)?);
            }
        }
    }

    // Add WITH clauses to the result
    result.with_clauses = with_clauses;

    Ok(result)
}

/// Converts a sqlparser WITH clause to our `WithClause` list.
fn convert_with_clause(with: sp::With) -> ParseResult<Vec<WithClause>> {
    let recursive = with.recursive;

    with.cte_tables
        .into_iter()
        .map(|cte| {
            let name = convert_ident(cte.alias.name);
            let columns: Vec<Identifier> =
                cte.alias.columns.into_iter().map(|col| convert_ident(col.name)).collect();
            let query = convert_query(*cte.query)?;

            Ok(WithClause { name, columns, query: Box::new(query), recursive })
        })
        .collect()
}

/// Converts a sqlparser Select to our `SelectStatement`.
fn convert_select(select: sp::Select) -> ParseResult<SelectStatement> {
    let distinct = match select.distinct {
        Some(sp::Distinct::Distinct) => true,
        Some(sp::Distinct::On(_)) => {
            return Err(ParseError::Unsupported("DISTINCT ON".to_string()))
        }
        None => false,
    };

    let projection =
        select.projection.into_iter().map(convert_select_item).collect::<ParseResult<Vec<_>>>()?;

    let from =
        select.from.into_iter().map(convert_table_with_joins).collect::<ParseResult<Vec<_>>>()?;

    let where_clause = select.selection.map(convert_expr).transpose()?;

    let group_by = match select.group_by {
        sp::GroupByExpr::Expressions(exprs, _) => {
            exprs.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?
        }
        sp::GroupByExpr::All(_) => return Err(ParseError::Unsupported("GROUP BY ALL".to_string())),
    };

    let having = select.having.map(convert_expr).transpose()?;

    // Parse named window definitions (WINDOW clause)
    let named_windows = select
        .named_window
        .into_iter()
        .map(convert_named_window)
        .collect::<ParseResult<Vec<_>>>()?;

    Ok(SelectStatement {
        with_clauses: vec![], // CTEs are handled at the Query level, not Select level
        distinct,
        projection,
        from,
        match_clause: None,             // Handled separately by extension parser
        optional_match_clauses: vec![], // Handled separately by extension parser
        mandatory_match: false,         // Handled separately by extension parser
        where_clause,
        group_by,
        having,
        named_windows,
        order_by: vec![],
        limit: None,
        offset: None,
        set_op: None,
    })
}

/// Converts a sqlparser `SelectItem`.
fn convert_select_item(item: sp::SelectItem) -> ParseResult<SelectItem> {
    match item {
        sp::SelectItem::UnnamedExpr(expr) => {
            Ok(SelectItem::Expr { expr: convert_expr(expr)?, alias: None })
        }
        sp::SelectItem::ExprWithAlias { expr, alias } => {
            Ok(SelectItem::Expr { expr: convert_expr(expr)?, alias: Some(convert_ident(alias)) })
        }
        sp::SelectItem::Wildcard(_) => Ok(SelectItem::Wildcard),
        sp::SelectItem::QualifiedWildcard(kind, _) => match kind {
            sp::SelectItemQualifiedWildcardKind::ObjectName(name) => {
                Ok(SelectItem::QualifiedWildcard(convert_object_name(name)))
            }
            sp::SelectItemQualifiedWildcardKind::Expr(_) => {
                Err(ParseError::Unsupported("qualified wildcard on expression".to_string()))
            }
        },
    }
}

/// Converts a table with joins.
fn convert_table_with_joins(twj: sp::TableWithJoins) -> ParseResult<TableRef> {
    let mut result = convert_table_factor(twj.relation)?;

    for join in twj.joins {
        let right = convert_table_factor(join.relation)?;
        let join_type = match join.join_operator {
            sp::JoinOperator::Inner(_) | sp::JoinOperator::Join(_) => JoinType::Inner,
            sp::JoinOperator::LeftOuter(_) | sp::JoinOperator::Left(_) => JoinType::LeftOuter,
            sp::JoinOperator::RightOuter(_) | sp::JoinOperator::Right(_) => JoinType::RightOuter,
            sp::JoinOperator::FullOuter(_) => JoinType::FullOuter,
            sp::JoinOperator::CrossJoin(_) => JoinType::Cross,
            sp::JoinOperator::LeftSemi(_)
            | sp::JoinOperator::RightSemi(_)
            | sp::JoinOperator::Semi(_) => {
                return Err(ParseError::Unsupported("SEMI JOIN".to_string()));
            }
            sp::JoinOperator::LeftAnti(_)
            | sp::JoinOperator::RightAnti(_)
            | sp::JoinOperator::Anti(_) => {
                return Err(ParseError::Unsupported("ANTI JOIN".to_string()));
            }
            sp::JoinOperator::AsOf { .. } => {
                return Err(ParseError::Unsupported("AS OF JOIN".to_string()));
            }
            sp::JoinOperator::CrossApply | sp::JoinOperator::OuterApply => {
                return Err(ParseError::Unsupported("APPLY".to_string()));
            }
            sp::JoinOperator::StraightJoin(_) => {
                return Err(ParseError::Unsupported("STRAIGHT JOIN".to_string()));
            }
        };

        let condition = match join.join_operator {
            sp::JoinOperator::Inner(constraint)
            | sp::JoinOperator::Join(constraint)
            | sp::JoinOperator::LeftOuter(constraint)
            | sp::JoinOperator::Left(constraint)
            | sp::JoinOperator::RightOuter(constraint)
            | sp::JoinOperator::Right(constraint)
            | sp::JoinOperator::FullOuter(constraint) => convert_join_constraint(constraint)?,
            // All other join types have no condition
            _ => JoinCondition::None,
        };

        result = TableRef::Join(Box::new(JoinClause { left: result, right, join_type, condition }));
    }

    Ok(result)
}

/// Converts a join constraint.
fn convert_join_constraint(constraint: sp::JoinConstraint) -> ParseResult<JoinCondition> {
    match constraint {
        sp::JoinConstraint::On(expr) => Ok(JoinCondition::On(convert_expr(expr)?)),
        sp::JoinConstraint::Using(names) => {
            // Extract first identifier from each ObjectName
            let idents = names
                .into_iter()
                .filter_map(|name| {
                    name.0.into_iter().next().map(|part| {
                        convert_ident(
                            part.as_ident().cloned().unwrap_or_else(|| sp::Ident::new("")),
                        )
                    })
                })
                .collect();
            Ok(JoinCondition::Using(idents))
        }
        sp::JoinConstraint::Natural => Ok(JoinCondition::Natural),
        sp::JoinConstraint::None => Ok(JoinCondition::None),
    }
}

/// Converts a table factor.
fn convert_table_factor(factor: sp::TableFactor) -> ParseResult<TableRef> {
    match factor {
        sp::TableFactor::Table { name, alias, .. } => Ok(TableRef::Table {
            name: convert_object_name(name),
            alias: alias.map(convert_table_alias),
        }),
        sp::TableFactor::Derived { subquery, alias, lateral } => {
            let alias =
                alias.ok_or_else(|| ParseError::MissingClause("alias for subquery".to_string()))?;
            let query = Box::new(convert_query(*subquery)?);
            let alias = convert_table_alias(alias);
            if lateral {
                Ok(TableRef::LateralSubquery { query, alias })
            } else {
                Ok(TableRef::Subquery { query, alias })
            }
        }
        sp::TableFactor::TableFunction { expr, alias } => {
            // Extract function name and args from the expression
            if let sp::Expr::Function(func) = expr {
                Ok(TableRef::TableFunction {
                    name: convert_object_name(func.name),
                    args: convert_function_args(func.args)?,
                    alias: alias.map(convert_table_alias),
                })
            } else {
                Err(ParseError::Unsupported("non-function table function".to_string()))
            }
        }
        sp::TableFactor::NestedJoin { table_with_joins, alias } => {
            let mut result = convert_table_with_joins(*table_with_joins)?;
            if let Some(alias) = alias {
                // Wrap in another table ref with alias if needed
                match &mut result {
                    TableRef::Table { alias: ref mut a, .. } => {
                        *a = Some(convert_table_alias(alias))
                    }
                    TableRef::Subquery { alias: ref mut a, .. }
                    | TableRef::LateralSubquery { alias: ref mut a, .. } => {
                        *a = convert_table_alias(alias)
                    }
                    _ => {}
                }
            }
            Ok(result)
        }
        _ => Err(ParseError::Unsupported("table factor type".to_string())),
    }
}

/// Converts function arguments.
fn convert_function_args(args: sp::FunctionArguments) -> ParseResult<Vec<Expr>> {
    match args {
        sp::FunctionArguments::None => Ok(vec![]),
        sp::FunctionArguments::Subquery(_) => {
            Err(ParseError::Unsupported("subquery function argument".to_string()))
        }
        sp::FunctionArguments::List(arg_list) => arg_list
            .args
            .into_iter()
            .map(|arg| match arg {
                sp::FunctionArg::Unnamed(expr) => expr,
                sp::FunctionArg::Named { arg, .. } => arg,
                sp::FunctionArg::ExprNamed { arg, .. } => arg,
            })
            .map(|arg_expr| match arg_expr {
                sp::FunctionArgExpr::Expr(e) => convert_expr(e),
                sp::FunctionArgExpr::QualifiedWildcard(name) => {
                    Ok(Expr::QualifiedWildcard(convert_object_name(name)))
                }
                sp::FunctionArgExpr::Wildcard => Ok(Expr::Wildcard),
            })
            .collect::<ParseResult<Vec<_>>>(),
    }
}

/// Converts a table alias.
fn convert_table_alias(alias: sp::TableAlias) -> TableAlias {
    TableAlias {
        name: convert_ident(alias.name),
        columns: alias.columns.into_iter().map(|col| convert_ident(col.name)).collect(),
    }
}

/// Converts an expression.
#[allow(clippy::too_many_lines)]
fn convert_expr(expr: sp::Expr) -> ParseResult<Expr> {
    match expr {
        sp::Expr::Identifier(ident) => {
            Ok(Expr::Column(QualifiedName::simple(convert_ident(ident))))
        }
        sp::Expr::CompoundIdentifier(idents) => {
            Ok(Expr::Column(QualifiedName::new(idents.into_iter().map(convert_ident).collect())))
        }
        sp::Expr::Value(value) => convert_value(value.value),
        sp::Expr::BinaryOp { left, op, right } => {
            let left = convert_expr(*left)?;
            let right = convert_expr(*right)?;
            let op = convert_binary_op(&op)?;
            Ok(Expr::BinaryOp { left: Box::new(left), op, right: Box::new(right) })
        }
        sp::Expr::UnaryOp { op, expr } => {
            let operand = convert_expr(*expr)?;
            let op = convert_unary_op(op)?;
            Ok(Expr::UnaryOp { op, operand: Box::new(operand) })
        }
        sp::Expr::Nested(inner) => convert_expr(*inner),
        sp::Expr::Function(func) => convert_function(func),
        sp::Expr::Cast { expr, data_type, .. } => Ok(Expr::Cast {
            expr: Box::new(convert_expr(*expr)?),
            data_type: format_data_type(&data_type),
        }),
        sp::Expr::Case { operand, conditions, else_result, .. } => {
            let when_clauses: Vec<(Expr, Expr)> = conditions
                .into_iter()
                .map(|case_when| {
                    Ok((convert_expr(case_when.condition)?, convert_expr(case_when.result)?))
                })
                .collect::<ParseResult<Vec<_>>>()?;

            Ok(Expr::Case(CaseExpr {
                operand: operand.map(|e| convert_expr(*e)).transpose()?.map(Box::new),
                when_clauses,
                else_result: else_result.map(|e| convert_expr(*e)).transpose()?.map(Box::new),
            }))
        }
        sp::Expr::Subquery(query) => Ok(Expr::Subquery(crate::ast::expr::Subquery {
            query: Box::new(convert_query(*query)?),
        })),
        sp::Expr::Exists { subquery, negated } => Ok(Expr::Exists {
            subquery: crate::ast::expr::Subquery { query: Box::new(convert_query(*subquery)?) },
            negated,
        }),
        sp::Expr::InList { expr, list, negated } => Ok(Expr::InList {
            expr: Box::new(convert_expr(*expr)?),
            list: list.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?,
            negated,
        }),
        sp::Expr::InSubquery { expr, subquery, negated } => Ok(Expr::InSubquery {
            expr: Box::new(convert_expr(*expr)?),
            subquery: crate::ast::expr::Subquery { query: Box::new(convert_query(*subquery)?) },
            negated,
        }),
        sp::Expr::Between { expr, low, high, negated } => Ok(Expr::Between {
            expr: Box::new(convert_expr(*expr)?),
            low: Box::new(convert_expr(*low)?),
            high: Box::new(convert_expr(*high)?),
            negated,
        }),
        sp::Expr::IsNull(expr) => {
            Ok(Expr::UnaryOp { op: UnaryOp::IsNull, operand: Box::new(convert_expr(*expr)?) })
        }
        sp::Expr::IsNotNull(expr) => {
            Ok(Expr::UnaryOp { op: UnaryOp::IsNotNull, operand: Box::new(convert_expr(*expr)?) })
        }
        sp::Expr::Tuple(exprs) => {
            Ok(Expr::Tuple(exprs.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?))
        }
        sp::Expr::Array(arr) => {
            let sp::Array { elem, .. } = arr;
            // Try to convert to a vector or multi-vector literal
            convert_array_expr(elem)
        }
        sp::Expr::CompoundFieldAccess { root, access_chain } => {
            // Handle array indexing - take the first subscript if it's an index
            let mut result = convert_expr(*root)?;
            for access in access_chain {
                match access {
                    sp::AccessExpr::Subscript(sp::Subscript::Index { index }) => {
                        result = Expr::ArrayIndex {
                            array: Box::new(result),
                            index: Box::new(convert_expr(index)?),
                        };
                    }
                    sp::AccessExpr::Subscript(sp::Subscript::Slice { .. }) => {
                        return Err(ParseError::Unsupported("subscript slice".to_string()));
                    }
                    sp::AccessExpr::Dot(field_expr) => {
                        // Field access like a.b - if the field is an identifier, append it
                        match (result, field_expr) {
                            (Expr::Column(qname), sp::Expr::Identifier(ident)) => {
                                let mut parts = qname.parts;
                                parts.push(convert_ident(ident));
                                result = Expr::Column(QualifiedName::new(parts));
                            }
                            _ => {
                                return Err(ParseError::Unsupported(
                                    "field access on expression".to_string(),
                                ));
                            }
                        }
                    }
                }
            }
            Ok(result)
        }
        sp::Expr::Like { negated, expr, pattern, escape_char: _, any: _ } => Ok(Expr::BinaryOp {
            left: Box::new(convert_expr(*expr)?),
            op: if negated { BinaryOp::NotLike } else { BinaryOp::Like },
            right: Box::new(convert_expr(*pattern)?),
        }),
        sp::Expr::ILike { negated, expr, pattern, escape_char: _, any: _ } => Ok(Expr::BinaryOp {
            left: Box::new(convert_expr(*expr)?),
            op: if negated { BinaryOp::NotILike } else { BinaryOp::ILike },
            right: Box::new(convert_expr(*pattern)?),
        }),
        sp::Expr::Named { name, .. } => {
            // Named parameter like $name
            Ok(Expr::Parameter(ParameterRef::Named(name.value)))
        }
        // TypedString: DATE '2024-01-15', TIME '10:30:00', TIMESTAMP '2024-01-15T10:30:00'
        sp::Expr::TypedString(typed_string) => {
            let sp::TypedString { data_type, value, .. } = typed_string;
            // Extract string value from ValueWithSpan
            let str_val = match value.value {
                sp::Value::SingleQuotedString(s) => s,
                sp::Value::DoubleQuotedString(s) => s,
                sp::Value::Number(n, _) => n,
                _ => return Err(ParseError::InvalidLiteral("expected string value".to_string())),
            };
            // Convert typed string to function call: DATE '...' -> TO_DATE('...')
            // or just return as a string literal with type info stored as a function call
            match &data_type {
                sp::DataType::Date => {
                    // DATE 'value' -> string literal that represents date
                    Ok(Expr::Function(FunctionCall {
                        name: QualifiedName::simple("date".to_string()),
                        args: vec![Expr::Literal(Literal::String(str_val))],
                        distinct: false,
                        filter: None,
                        over: None,
                    }))
                }
                sp::DataType::Time(..) => {
                    // TIME 'value' -> time function call
                    Ok(Expr::Function(FunctionCall {
                        name: QualifiedName::simple("time".to_string()),
                        args: vec![Expr::Literal(Literal::String(str_val))],
                        distinct: false,
                        filter: None,
                        over: None,
                    }))
                }
                sp::DataType::Timestamp(..) => {
                    // TIMESTAMP 'value' -> datetime function call
                    Ok(Expr::Function(FunctionCall {
                        name: QualifiedName::simple("datetime".to_string()),
                        args: vec![Expr::Literal(Literal::String(str_val))],
                        distinct: false,
                        filter: None,
                        over: None,
                    }))
                }
                _ => {
                    // For other typed strings, just return the string value
                    Ok(Expr::Literal(Literal::String(str_val)))
                }
            }
        }
        // INTERVAL '1 day', INTERVAL '2 hours', etc.
        sp::Expr::Interval(interval) => {
            // Convert interval to duration function call
            // INTERVAL 'P1D' or INTERVAL '1 day' -> duration('P1D')
            let value = match *interval.value {
                sp::Expr::Value(sp::ValueWithSpan {
                    value: sp::Value::SingleQuotedString(s),
                    ..
                }) => s,
                sp::Expr::Value(sp::ValueWithSpan {
                    value: sp::Value::DoubleQuotedString(s),
                    ..
                }) => s,
                sp::Expr::Value(sp::ValueWithSpan { value: sp::Value::Number(n, _), .. }) => {
                    // If just a number, try to construct with leading_field
                    let unit = interval
                        .leading_field
                        .map(|f| format!("{:?}", f).to_lowercase())
                        .unwrap_or_else(|| "day".to_string());
                    format!("{} {}", n, unit)
                }
                other => {
                    // For other expression types, convert and return directly
                    return convert_expr(other);
                }
            };

            // Convert SQL interval format to ISO 8601 duration
            let duration = convert_sql_interval_to_iso8601(&value);

            Ok(Expr::Function(FunctionCall {
                name: QualifiedName::simple("duration".to_string()),
                args: vec![Expr::Literal(Literal::String(duration))],
                distinct: false,
                filter: None,
                over: None,
            }))
        }
        // Handle placeholder for positional parameters
        _ => Err(ParseError::Unsupported(format!("expression type: {expr:?}"))),
    }
}

/// Converts SQL interval format to ISO 8601 duration format.
///
/// Examples:
/// - '1 day' -> 'P1D'
/// - '2 hours' -> 'PT2H'
/// - '1 year 2 months' -> 'P1Y2M'
/// - '1 day 2 hours 30 minutes' -> 'P1DT2H30M'
fn convert_sql_interval_to_iso8601(value: &str) -> String {
    // If already in ISO 8601 format, return as-is
    if value.starts_with('P') {
        return value.to_string();
    }

    let value_lower = value.to_lowercase();
    let mut years = 0i64;
    let mut months = 0i64;
    let mut days = 0i64;
    let mut hours = 0i64;
    let mut minutes = 0i64;
    let mut seconds = 0i64;

    // Parse "N unit" pairs
    let mut current_num: Option<i64> = None;
    for token in value_lower.split_whitespace() {
        if let Ok(num) = token.parse::<i64>() {
            current_num = Some(num);
        } else if let Some(num) = current_num.take() {
            match token.trim_end_matches('s') {
                "year" => years = num,
                "month" => months = num,
                "week" => days += num * 7,
                "day" => days = num,
                "hour" => hours = num,
                "minute" | "min" => minutes = num,
                "second" | "sec" => seconds = num,
                _ => {}
            }
        }
    }

    use std::fmt::Write;

    // Build ISO 8601 duration string
    let mut result = String::from("P");

    if years > 0 {
        let _ = write!(result, "{years}Y");
    }
    if months > 0 {
        let _ = write!(result, "{months}M");
    }
    if days > 0 {
        let _ = write!(result, "{days}D");
    }

    if hours > 0 || minutes > 0 || seconds > 0 {
        result.push('T');
        if hours > 0 {
            let _ = write!(result, "{hours}H");
        }
        if minutes > 0 {
            let _ = write!(result, "{minutes}M");
        }
        if seconds > 0 {
            let _ = write!(result, "{seconds}S");
        }
    }

    if result == "P" {
        // No values parsed, return original
        return value.to_string();
    }

    result
}

/// Converts an array expression, detecting vector and multi-vector literals.
///
/// This function analyzes the array elements to determine if they form:
/// - A vector literal: `[0.1, 0.2, 0.3]` -> `Literal::Vector`
/// - A multi-vector literal: `[[0.1, 0.2], [0.3, 0.4]]` -> `Literal::MultiVector`
/// - A general tuple/array for other cases
fn convert_array_expr(elements: Vec<sp::Expr>) -> ParseResult<Expr> {
    // Check if all elements are numeric literals (for vector)
    let all_numeric =
        elements.iter().all(|e| matches!(e, sp::Expr::Value(v) if is_numeric_value(v)));

    if all_numeric && !elements.is_empty() {
        // Convert to a vector literal
        let values: Vec<f32> = elements
            .iter()
            .map(|e| {
                if let sp::Expr::Value(v) = e {
                    value_to_f32(v)
                } else {
                    Err(ParseError::InvalidLiteral("expected numeric value".to_string()))
                }
            })
            .collect::<ParseResult<Vec<_>>>()?;
        return Ok(Expr::Literal(Literal::Vector(values)));
    }

    // Check if all elements are arrays of numeric literals (for multi-vector)
    let all_arrays = elements.iter().all(|e| {
        matches!(e, sp::Expr::Array(arr) if arr.elem.iter().all(|inner| matches!(inner, sp::Expr::Value(v) if is_numeric_value(v))))
    });

    if all_arrays && !elements.is_empty() {
        // Convert to a multi-vector literal
        let vectors: Vec<Vec<f32>> = elements
            .iter()
            .map(|e| {
                if let sp::Expr::Array(arr) = e {
                    arr.elem
                        .iter()
                        .map(|inner| {
                            if let sp::Expr::Value(v) = inner {
                                value_to_f32(v)
                            } else {
                                Err(ParseError::InvalidLiteral(
                                    "expected numeric value in nested array".to_string(),
                                ))
                            }
                        })
                        .collect::<ParseResult<Vec<_>>>()
                } else {
                    Err(ParseError::InvalidLiteral("expected array in multi-vector".to_string()))
                }
            })
            .collect::<ParseResult<Vec<_>>>()?;
        return Ok(Expr::Literal(Literal::MultiVector(vectors)));
    }

    // Fall back to Tuple for other cases
    let converted = elements.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?;
    Ok(Expr::Tuple(converted))
}

/// Checks if a sqlparser Value is a numeric literal.
fn is_numeric_value(value: &sp::ValueWithSpan) -> bool {
    matches!(value.value, sp::Value::Number(_, _))
}

/// Converts a sqlparser Value to f32.
fn value_to_f32(value: &sp::ValueWithSpan) -> ParseResult<f32> {
    match &value.value {
        sp::Value::Number(n, _) => {
            n.parse::<f32>().map_err(|_| ParseError::InvalidLiteral(format!("invalid f32: {n}")))
        }
        _ => Err(ParseError::InvalidLiteral("expected numeric value".to_string())),
    }
}

/// Converts a sqlparser Value to our Expr.
fn convert_value(value: sp::Value) -> ParseResult<Expr> {
    match value {
        sp::Value::Null => Ok(Expr::Literal(Literal::Null)),
        sp::Value::Boolean(b) => Ok(Expr::Literal(Literal::Boolean(b))),
        sp::Value::Number(n, _) => {
            // Try to parse as integer first, then float
            if let Ok(i) = n.parse::<i64>() {
                Ok(Expr::Literal(Literal::Integer(i)))
            } else if let Ok(f) = n.parse::<f64>() {
                Ok(Expr::Literal(Literal::Float(f)))
            } else {
                Err(ParseError::InvalidLiteral(format!("invalid number: {n}")))
            }
        }
        sp::Value::SingleQuotedString(s) | sp::Value::DoubleQuotedString(s) => {
            Ok(Expr::Literal(Literal::String(s)))
        }
        sp::Value::Placeholder(p) => {
            if p == "?" {
                Ok(Expr::Parameter(ParameterRef::Anonymous))
            } else if let Some(n) = p.strip_prefix('$') {
                if let Ok(pos) = n.parse::<u32>() {
                    Ok(Expr::Parameter(ParameterRef::Positional(pos)))
                } else {
                    Ok(Expr::Parameter(ParameterRef::Named(n.to_string())))
                }
            } else {
                Err(ParseError::InvalidLiteral(format!("unknown placeholder: {p}")))
            }
        }
        _ => Err(ParseError::Unsupported(format!("value type: {value:?}"))),
    }
}

/// Converts a binary operator.
fn convert_binary_op(op: &sp::BinaryOperator) -> ParseResult<BinaryOp> {
    match op {
        sp::BinaryOperator::Plus => Ok(BinaryOp::Add),
        sp::BinaryOperator::Minus => Ok(BinaryOp::Sub),
        sp::BinaryOperator::Multiply => Ok(BinaryOp::Mul),
        sp::BinaryOperator::Divide => Ok(BinaryOp::Div),
        sp::BinaryOperator::Modulo => Ok(BinaryOp::Mod),
        sp::BinaryOperator::Eq => Ok(BinaryOp::Eq),
        sp::BinaryOperator::NotEq => Ok(BinaryOp::NotEq),
        sp::BinaryOperator::Lt => Ok(BinaryOp::Lt),
        sp::BinaryOperator::LtEq => Ok(BinaryOp::LtEq),
        sp::BinaryOperator::Gt => Ok(BinaryOp::Gt),
        sp::BinaryOperator::GtEq => Ok(BinaryOp::GtEq),
        sp::BinaryOperator::And => Ok(BinaryOp::And),
        sp::BinaryOperator::Or => Ok(BinaryOp::Or),
        // Custom operators for vector operations (will be handled by extension parser)
        sp::BinaryOperator::Arrow => Err(ParseError::Unsupported("-> operator".to_string())),
        sp::BinaryOperator::LongArrow => Err(ParseError::Unsupported("->> operator".to_string())),
        sp::BinaryOperator::HashArrow => Err(ParseError::Unsupported("#> operator".to_string())),
        sp::BinaryOperator::HashLongArrow => {
            Err(ParseError::Unsupported("#>> operator".to_string()))
        }
        _ => Err(ParseError::Unsupported(format!("binary operator: {op:?}"))),
    }
}

/// Converts a unary operator.
fn convert_unary_op(op: sp::UnaryOperator) -> ParseResult<UnaryOp> {
    match op {
        sp::UnaryOperator::Not => Ok(UnaryOp::Not),
        // Unary plus is treated as a no-op (identity), but we convert it to Neg
        // with a special case since there's no identity op - the caller should
        // handle this by not wrapping in UnaryOp at all for plus
        sp::UnaryOperator::Plus | sp::UnaryOperator::Minus => Ok(UnaryOp::Neg),
        _ => Err(ParseError::Unsupported(format!("unary operator: {op:?}"))),
    }
}

/// Converts a function call.
fn convert_function(func: sp::Function) -> ParseResult<Expr> {
    let name = convert_object_name(func.name);
    let args = convert_function_args(func.args)?;

    let filter = func.filter.map(|f| convert_expr(*f)).transpose()?.map(Box::new);

    let over = func.over.map(convert_window_spec).transpose()?;

    Ok(Expr::Function(FunctionCall {
        name,
        args,
        distinct: false, // sqlparser 0.52 handles this differently
        filter,
        over,
    }))
}

/// Converts a CALL statement.
///
/// Note: This converts basic CALL statements. YIELD clause support
/// is handled separately by the ExtendedParser.
fn convert_call(func: sp::Function) -> ParseResult<CallStatement> {
    let procedure_name = convert_object_name(func.name);
    let arguments = convert_function_args(func.args)?;

    Ok(CallStatement::new(procedure_name, arguments))
}

/// Converts a window specification.
fn convert_window_spec(spec: sp::WindowType) -> ParseResult<WindowSpec> {
    match spec {
        sp::WindowType::WindowSpec(spec) => {
            // Handle reference to a named window
            let window_name = spec.window_name.map(convert_ident);

            let partition_by =
                spec.partition_by.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?;

            let order_by = spec
                .order_by
                .into_iter()
                .map(convert_order_by_expr)
                .collect::<ParseResult<Vec<_>>>()?;

            let frame = spec.window_frame.map(convert_window_frame).transpose()?;

            Ok(WindowSpec { window_name, partition_by, order_by, frame })
        }
        sp::WindowType::NamedWindow(name) => {
            // Direct reference to a named window (e.g., `OVER w`)
            Ok(WindowSpec {
                window_name: Some(convert_ident(name)),
                partition_by: vec![],
                order_by: vec![],
                frame: None,
            })
        }
    }
}

/// Converts a named window definition (from WINDOW clause).
fn convert_named_window(def: sp::NamedWindowDefinition) -> ParseResult<NamedWindowDefinition> {
    let name = convert_ident(def.0);
    match def.1 {
        sp::NamedWindowExpr::NamedWindow(base_name) => {
            // Window that references another named window: WINDOW w AS prev_window
            Ok(NamedWindowDefinition {
                name,
                base_window: Some(convert_ident(base_name)),
                spec: WindowSpec {
                    window_name: None,
                    partition_by: vec![],
                    order_by: vec![],
                    frame: None,
                },
            })
        }
        sp::NamedWindowExpr::WindowSpec(spec) => {
            // Window with full specification: WINDOW w AS (PARTITION BY ...)
            let window_name = spec.window_name.map(convert_ident);
            let partition_by =
                spec.partition_by.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?;
            let order_by = spec
                .order_by
                .into_iter()
                .map(convert_order_by_expr)
                .collect::<ParseResult<Vec<_>>>()?;
            let frame = spec.window_frame.map(convert_window_frame).transpose()?;

            Ok(NamedWindowDefinition {
                name,
                base_window: window_name,
                spec: WindowSpec { window_name: None, partition_by, order_by, frame },
            })
        }
    }
}

/// Converts a window frame.
fn convert_window_frame(frame: sp::WindowFrame) -> ParseResult<WindowFrame> {
    let units = match frame.units {
        sp::WindowFrameUnits::Rows => WindowFrameUnits::Rows,
        sp::WindowFrameUnits::Range => WindowFrameUnits::Range,
        sp::WindowFrameUnits::Groups => WindowFrameUnits::Groups,
    };

    let start = convert_window_frame_bound(frame.start_bound)?;
    let end = frame.end_bound.map(convert_window_frame_bound).transpose()?;

    // Note: sqlparser doesn't support frame exclusion yet (TBD in their codebase)
    // We'll set it to None for now and add support when sqlparser implements it
    Ok(WindowFrame { units, start, end, exclusion: None })
}

/// Converts a window frame bound.
fn convert_window_frame_bound(bound: sp::WindowFrameBound) -> ParseResult<WindowFrameBound> {
    match bound {
        sp::WindowFrameBound::CurrentRow => Ok(WindowFrameBound::CurrentRow),
        sp::WindowFrameBound::Preceding(None) => Ok(WindowFrameBound::UnboundedPreceding),
        sp::WindowFrameBound::Following(None) => Ok(WindowFrameBound::UnboundedFollowing),
        sp::WindowFrameBound::Preceding(Some(expr)) => {
            Ok(WindowFrameBound::Preceding(Box::new(convert_expr(*expr)?)))
        }
        sp::WindowFrameBound::Following(Some(expr)) => {
            Ok(WindowFrameBound::Following(Box::new(convert_expr(*expr)?)))
        }
    }
}

/// Converts an ORDER BY expression.
fn convert_order_by_expr(expr: sp::OrderByExpr) -> ParseResult<OrderByExpr> {
    let asc = expr.options.asc.unwrap_or(true); // Default to ASC

    Ok(OrderByExpr {
        expr: Box::new(convert_expr(expr.expr)?),
        asc,
        nulls_first: expr.options.nulls_first,
    })
}

/// Converts an INSERT statement.
fn convert_insert(insert: sp::Insert) -> ParseResult<InsertStatement> {
    // Extract table name from TableObject
    let table = match insert.table {
        sp::TableObject::TableName(name) => convert_object_name(name),
        sp::TableObject::TableFunction(_) => {
            return Err(ParseError::Unsupported("INSERT into table function".to_string()));
        }
    };

    let columns: Vec<Identifier> = insert.columns.into_iter().map(convert_ident).collect();

    let source = match insert.source {
        Some(source) => match *source.body {
            sp::SetExpr::Values(values) => {
                let rows: Vec<Vec<Expr>> = values
                    .rows
                    .into_iter()
                    .map(|row| row.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>())
                    .collect::<ParseResult<Vec<_>>>()?;
                InsertSource::Values(rows)
            }
            sp::SetExpr::Select(select) => {
                let query = convert_select(*select)?;
                InsertSource::Query(Box::new(query))
            }
            _ => return Err(ParseError::Unsupported("INSERT source type".to_string())),
        },
        None => InsertSource::DefaultValues,
    };

    let on_conflict = insert.on.map(convert_on_conflict).transpose()?;

    let returning = insert
        .returning
        .map(|items| items.into_iter().map(convert_select_item).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    Ok(InsertStatement { table, columns, source, on_conflict, returning })
}

/// Converts ON CONFLICT clause.
fn convert_on_conflict(on: sp::OnInsert) -> ParseResult<OnConflict> {
    match on {
        sp::OnInsert::DuplicateKeyUpdate(assignments) => {
            Ok(OnConflict {
                target: ConflictTarget::Columns(vec![]), // MySQL doesn't specify columns
                action: ConflictAction::DoUpdate {
                    assignments: assignments
                        .into_iter()
                        .map(convert_assignment)
                        .collect::<ParseResult<Vec<_>>>()?,
                    where_clause: None,
                },
            })
        }
        sp::OnInsert::OnConflict(conflict) => {
            let target = match conflict.conflict_target {
                Some(sp::ConflictTarget::Columns(cols)) => {
                    ConflictTarget::Columns(cols.into_iter().map(convert_ident).collect())
                }
                Some(sp::ConflictTarget::OnConstraint(name)) => {
                    let converted = convert_object_name(name);
                    let ident = converted.parts.into_iter().next().ok_or_else(|| {
                        ParseError::MissingClause("constraint name in ON CONFLICT".to_string())
                    })?;
                    ConflictTarget::Constraint(ident)
                }
                None => ConflictTarget::Columns(vec![]),
            };

            let action = match conflict.action {
                sp::OnConflictAction::DoNothing => ConflictAction::DoNothing,
                sp::OnConflictAction::DoUpdate(update) => ConflictAction::DoUpdate {
                    assignments: update
                        .assignments
                        .into_iter()
                        .map(convert_assignment)
                        .collect::<ParseResult<Vec<_>>>()?,
                    where_clause: update.selection.map(convert_expr).transpose()?,
                },
            };

            Ok(OnConflict { target, action })
        }
        _ => Err(ParseError::Unsupported("ON INSERT type".to_string())),
    }
}

/// Converts an ObjectNamePart to an Ident.
fn object_name_part_to_ident(part: sp::ObjectNamePart) -> Option<sp::Ident> {
    match part {
        sp::ObjectNamePart::Identifier(ident) => Some(ident),
        sp::ObjectNamePart::Function(_) => None,
    }
}

/// Converts an assignment (for UPDATE or ON CONFLICT).
fn convert_assignment(assign: sp::Assignment) -> ParseResult<Assignment> {
    // Convert assignment target to column name
    let column = match assign.target {
        sp::AssignmentTarget::ColumnName(names) => names
            .0
            .into_iter()
            .next()
            .and_then(object_name_part_to_ident)
            .map(convert_ident)
            .ok_or_else(|| ParseError::MissingClause("assignment target".to_string()))?,
        sp::AssignmentTarget::Tuple(_) => {
            return Err(ParseError::Unsupported("tuple assignment target".to_string()));
        }
    };

    Ok(Assignment { column, value: convert_expr(assign.value)? })
}

/// Converts an UPDATE statement.
fn convert_update(
    table: sp::TableWithJoins,
    assignments: Vec<sp::Assignment>,
    from: Option<Vec<sp::TableWithJoins>>,
    selection: Option<sp::Expr>,
    returning: Option<Vec<sp::SelectItem>>,
) -> ParseResult<UpdateStatement> {
    let table_ref = convert_table_with_joins(table)?;
    let TableRef::Table { name: table_name, alias } = table_ref else {
        return Err(ParseError::Unsupported("complex UPDATE target".to_string()));
    };

    let assignments =
        assignments.into_iter().map(convert_assignment).collect::<ParseResult<Vec<_>>>()?;

    let from_clause = from
        .map(|f| f.into_iter().map(convert_table_with_joins).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    let where_clause = selection.map(convert_expr).transpose()?;

    let returning = returning
        .map(|items| items.into_iter().map(convert_select_item).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    Ok(UpdateStatement {
        table: table_name,
        alias,
        assignments,
        from: from_clause,
        match_clause: None,
        where_clause,
        returning,
    })
}

/// Converts a DELETE statement.
fn convert_delete(delete: sp::Delete) -> ParseResult<DeleteStatement> {
    let from_table = match delete.from {
        sp::FromTable::WithFromKeyword(tables) => tables
            .into_iter()
            .next()
            .ok_or_else(|| ParseError::MissingClause("FROM".to_string()))?,
        sp::FromTable::WithoutKeyword(tables) => tables
            .into_iter()
            .next()
            .ok_or_else(|| ParseError::MissingClause("table".to_string()))?,
    };

    let table_ref = convert_table_with_joins(from_table)?;
    let TableRef::Table { name: table_name, alias } = table_ref else {
        return Err(ParseError::Unsupported("complex DELETE target".to_string()));
    };

    let using = delete
        .using
        .map(|u| u.into_iter().map(convert_table_with_joins).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    let where_clause = delete.selection.map(convert_expr).transpose()?;

    let returning = delete
        .returning
        .map(|items| items.into_iter().map(convert_select_item).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    Ok(DeleteStatement {
        table: table_name,
        alias,
        using,
        match_clause: None,
        where_clause,
        returning,
    })
}

/// Converts a CREATE TABLE statement.
fn convert_create_table(create: sp::CreateTable) -> ParseResult<CreateTableStatement> {
    let columns =
        create.columns.into_iter().map(convert_column_def).collect::<ParseResult<Vec<_>>>()?;

    let constraints = create
        .constraints
        .into_iter()
        .map(convert_table_constraint)
        .collect::<ParseResult<Vec<_>>>()?;

    // Handle PARTITION BY clause
    let partition_by = create.partition_by.map(|pb| convert_partition_by(pb)).transpose()?;

    // Handle PARTITION OF clause (sqlparser uses `clone` field for CLONE, not PARTITION OF)
    // PARTITION OF parsing requires extended parser support
    let partition_of: Option<PartitionOf> = None;

    Ok(CreateTableStatement {
        if_not_exists: create.if_not_exists,
        name: convert_object_name(create.name),
        columns,
        constraints,
        partition_by,
        partition_of,
    })
}

/// Converts a PARTITION BY clause.
fn convert_partition_by(pb: Box<sp::Expr>) -> ParseResult<PartitionBy> {
    // sqlparser represents PARTITION BY as an expression
    // We need to extract the partition strategy and columns
    // This is a simplified implementation - actual parsing may need more work
    match *pb {
        sp::Expr::Function(func) => {
            let name = func.name.to_string().to_uppercase();
            let columns = convert_function_args(func.args)?;
            let exprs: Vec<Expr> = columns;

            match name.as_str() {
                "RANGE" => Ok(PartitionBy::Range { columns: exprs }),
                "LIST" => Ok(PartitionBy::List { columns: exprs }),
                "HASH" => Ok(PartitionBy::Hash { columns: exprs }),
                _ => Err(ParseError::Unsupported(format!("Partition strategy: {name}"))),
            }
        }
        _ => Err(ParseError::Unsupported("Complex PARTITION BY expression".to_string())),
    }
}

/// Converts an ALTER INDEX statement.
fn convert_alter_index(
    name: sp::ObjectName,
    operation: sp::AlterIndexOperation,
) -> ParseResult<AlterIndexStatement> {
    let index_name = convert_object_name(name);
    let action = match operation {
        sp::AlterIndexOperation::RenameIndex { index_name: new_name } => {
            let new_ident = convert_object_name(new_name)
                .parts
                .into_iter()
                .next()
                .ok_or_else(|| ParseError::MissingClause("new index name".to_string()))?;
            AlterIndexAction::RenameIndex { new_name: new_ident }
        }
    };

    Ok(AlterIndexStatement { if_exists: false, name: index_name, action })
}

/// Converts a TRUNCATE statement.
fn convert_truncate(truncate: sp::Truncate) -> ParseResult<TruncateTableStatement> {
    let names: Vec<QualifiedName> =
        truncate.table_names.into_iter().map(|t| convert_object_name(t.name)).collect();

    let identity = truncate.identity.map(|i| match i {
        sp::TruncateIdentityOption::Restart => TruncateIdentity::Restart,
        sp::TruncateIdentityOption::Continue => TruncateIdentity::Continue,
    });

    let cascade = truncate.cascade.map(|c| match c {
        sp::CascadeOption::Cascade => TruncateCascade::Cascade,
        sp::CascadeOption::Restrict => TruncateCascade::Restrict,
    });

    Ok(TruncateTableStatement { names, identity, cascade })
}

/// Converts a column definition.
fn convert_column_def(col: sp::ColumnDef) -> ParseResult<ColumnDef> {
    let constraints =
        col.options.into_iter().filter_map(|opt| convert_column_option(opt.option).ok()).collect();

    Ok(ColumnDef {
        name: convert_ident(col.name),
        data_type: convert_data_type(col.data_type)?,
        constraints,
    })
}

/// Converts a column option to a constraint.
fn convert_column_option(opt: sp::ColumnOption) -> ParseResult<ColumnConstraint> {
    match opt {
        sp::ColumnOption::Null => Ok(ColumnConstraint::Null),
        sp::ColumnOption::NotNull => Ok(ColumnConstraint::NotNull),
        sp::ColumnOption::PrimaryKey(_) => Ok(ColumnConstraint::PrimaryKey),
        sp::ColumnOption::Unique(_) => Ok(ColumnConstraint::Unique),
        sp::ColumnOption::ForeignKey(fk) => Ok(ColumnConstraint::References {
            table: convert_object_name(fk.foreign_table),
            column: fk.referred_columns.into_iter().next().map(convert_ident),
        }),
        sp::ColumnOption::Check(check) => Ok(ColumnConstraint::Check(convert_expr(*check.expr)?)),
        sp::ColumnOption::Default(expr) => Ok(ColumnConstraint::Default(convert_expr(expr)?)),
        _ => Err(ParseError::Unsupported("column option".to_string())),
    }
}

/// Converts a table constraint.
fn convert_table_constraint(constraint: sp::TableConstraint) -> ParseResult<TableConstraint> {
    match constraint {
        sp::TableConstraint::PrimaryKey(pk) => Ok(TableConstraint::PrimaryKey {
            name: pk.name.map(convert_ident),
            columns: pk.columns.into_iter().map(convert_index_column_to_ident).collect(),
        }),
        sp::TableConstraint::Unique(unique) => Ok(TableConstraint::Unique {
            name: unique.name.map(convert_ident),
            columns: unique.columns.into_iter().map(convert_index_column_to_ident).collect(),
        }),
        sp::TableConstraint::ForeignKey(fk) => Ok(TableConstraint::ForeignKey {
            name: fk.name.map(convert_ident),
            columns: fk.columns.into_iter().map(convert_ident).collect(),
            references_table: convert_object_name(fk.foreign_table),
            references_columns: fk.referred_columns.into_iter().map(convert_ident).collect(),
        }),
        sp::TableConstraint::Check(check) => Ok(TableConstraint::Check {
            name: check.name.map(convert_ident),
            expr: convert_expr(*check.expr)?,
        }),
        _ => Err(ParseError::Unsupported("table constraint".to_string())),
    }
}

/// Converts an IndexColumn to an Identifier (extracting the column name).
fn convert_index_column_to_ident(col: sp::IndexColumn) -> Identifier {
    match col.column.expr {
        sp::Expr::Identifier(ident) => convert_ident(ident),
        _ => Identifier::new(format!("{}", col.column.expr)), // Fallback to string representation
    }
}

/// Converts a CREATE INDEX statement.
fn convert_create_index(create: sp::CreateIndex) -> ParseResult<CreateIndexStatement> {
    let name = create
        .name
        .map(convert_object_name)
        .and_then(|n| n.parts.into_iter().next())
        .ok_or_else(|| ParseError::MissingClause("index name".to_string()))?;

    let table = convert_object_name(create.table_name);

    let columns = create
        .columns
        .into_iter()
        .map(|col| {
            // IndexColumn now has a `column: OrderByExpr` field
            Ok(IndexColumn {
                expr: convert_expr(col.column.expr)?,
                asc: col.column.options.asc,
                nulls_first: col.column.options.nulls_first,
                opclass: None,
            })
        })
        .collect::<ParseResult<Vec<_>>>()?;

    Ok(CreateIndexStatement {
        unique: create.unique,
        if_not_exists: create.if_not_exists,
        name,
        table,
        columns,
        using: create.using.map(|idx_type| format!("{idx_type}")),
        with: vec![],
        where_clause: create.predicate.map(convert_expr).transpose()?,
    })
}

/// Converts a CREATE VIEW statement.
fn convert_create_view(
    or_replace: bool,
    name: sp::ObjectName,
    columns: Vec<sp::ViewColumnDef>,
    query: sp::Query,
) -> ParseResult<CreateViewStatement> {
    let view_name = convert_object_name(name);
    let column_aliases: Vec<Identifier> =
        columns.into_iter().map(|c| convert_ident(c.name)).collect();
    let select = convert_query(query)?;

    Ok(CreateViewStatement {
        or_replace,
        name: view_name,
        columns: column_aliases,
        query: Box::new(select),
    })
}

/// Converts an ALTER TABLE statement.
fn convert_alter_table(
    name: sp::ObjectName,
    if_exists: bool,
    operations: Vec<sp::AlterTableOperation>,
) -> ParseResult<AlterTableStatement> {
    let actions = operations
        .into_iter()
        .map(convert_alter_table_operation)
        .collect::<ParseResult<Vec<_>>>()?;

    Ok(AlterTableStatement { if_exists, name: convert_object_name(name), actions })
}

/// Converts an ALTER TABLE operation to our AlterTableAction.
fn convert_alter_table_operation(op: sp::AlterTableOperation) -> ParseResult<AlterTableAction> {
    match op {
        sp::AlterTableOperation::AddColumn { column_def, if_not_exists, .. } => {
            Ok(AlterTableAction::AddColumn {
                if_not_exists,
                column: convert_column_def(column_def)?,
            })
        }
        sp::AlterTableOperation::DropColumn { column_names, if_exists, drop_behavior, .. } => {
            // Take the first column name
            let column_name = column_names
                .into_iter()
                .next()
                .ok_or_else(|| ParseError::MissingClause("column name".to_string()))?;
            let cascade = matches!(drop_behavior, Some(sp::DropBehavior::Cascade));
            Ok(AlterTableAction::DropColumn {
                if_exists,
                column_name: convert_ident(column_name),
                cascade,
            })
        }
        sp::AlterTableOperation::AlterColumn { column_name, op } => {
            let action = convert_alter_column_op(op)?;
            Ok(AlterTableAction::AlterColumn { column_name: convert_ident(column_name), action })
        }
        sp::AlterTableOperation::RenameColumn { old_column_name, new_column_name } => {
            Ok(AlterTableAction::RenameColumn {
                old_name: convert_ident(old_column_name),
                new_name: convert_ident(new_column_name),
            })
        }
        sp::AlterTableOperation::RenameTable { table_name } => {
            let name = match table_name {
                sp::RenameTableNameKind::As(name) | sp::RenameTableNameKind::To(name) => name,
            };
            Ok(AlterTableAction::RenameTable { new_name: convert_object_name(name) })
        }
        sp::AlterTableOperation::AddConstraint { constraint, .. } => {
            Ok(AlterTableAction::AddConstraint(convert_table_constraint(constraint)?))
        }
        sp::AlterTableOperation::DropConstraint { name, if_exists, drop_behavior, .. } => {
            let cascade = matches!(drop_behavior, Some(sp::DropBehavior::Cascade));
            Ok(AlterTableAction::DropConstraint {
                if_exists,
                constraint_name: convert_ident(name),
                cascade,
            })
        }
        _ => Err(ParseError::Unsupported(format!("ALTER TABLE operation: {op:?}"))),
    }
}

/// Converts an ALTER COLUMN operation to our AlterColumnAction.
fn convert_alter_column_op(op: sp::AlterColumnOperation) -> ParseResult<AlterColumnAction> {
    match op {
        sp::AlterColumnOperation::SetNotNull => Ok(AlterColumnAction::SetNotNull),
        sp::AlterColumnOperation::DropNotNull => Ok(AlterColumnAction::DropNotNull),
        sp::AlterColumnOperation::SetDefault { value } => {
            Ok(AlterColumnAction::SetDefault(convert_expr(value)?))
        }
        sp::AlterColumnOperation::DropDefault => Ok(AlterColumnAction::DropDefault),
        sp::AlterColumnOperation::SetDataType { data_type, using, .. } => {
            Ok(AlterColumnAction::SetType {
                data_type: convert_data_type(data_type)?,
                using: using.map(convert_expr).transpose()?,
            })
        }
        _ => Err(ParseError::Unsupported(format!("ALTER COLUMN operation: {op:?}"))),
    }
}

/// Converts a data type.
#[allow(clippy::cast_possible_truncation)]
fn convert_data_type(dt: sp::DataType) -> ParseResult<DataType> {
    match dt {
        sp::DataType::Boolean | sp::DataType::Bool => Ok(DataType::Boolean),
        sp::DataType::SmallInt(_) | sp::DataType::Int2(_) => Ok(DataType::SmallInt),
        sp::DataType::Int(_) | sp::DataType::Integer(_) | sp::DataType::Int4(_) => {
            Ok(DataType::Integer)
        }
        sp::DataType::BigInt(_) | sp::DataType::Int8(_) => Ok(DataType::BigInt),
        sp::DataType::Real | sp::DataType::Float4 => Ok(DataType::Real),
        sp::DataType::DoublePrecision | sp::DataType::Double(_) | sp::DataType::Float8 => {
            Ok(DataType::DoublePrecision)
        }
        sp::DataType::Numeric(info) | sp::DataType::Decimal(info) => {
            let (precision, scale) = match info {
                sp::ExactNumberInfo::None => (None, None),
                sp::ExactNumberInfo::Precision(p) => (Some(p as u32), None),
                sp::ExactNumberInfo::PrecisionAndScale(p, s) => (Some(p as u32), Some(s as u32)),
            };
            Ok(DataType::Numeric { precision, scale })
        }
        sp::DataType::Varchar(len) | sp::DataType::CharVarying(len) => {
            let len_val = len.and_then(|l| match l {
                sp::CharacterLength::IntegerLength { length, .. } => Some(length as u32),
                sp::CharacterLength::Max => None,
            });
            Ok(DataType::Varchar(len_val))
        }
        sp::DataType::Text => Ok(DataType::Text),
        sp::DataType::Bytea => Ok(DataType::Bytea),
        sp::DataType::Timestamp(_, _) => Ok(DataType::Timestamp),
        sp::DataType::Date => Ok(DataType::Date),
        sp::DataType::Time(_, _) => Ok(DataType::Time),
        sp::DataType::Interval { .. } => Ok(DataType::Interval),
        sp::DataType::JSON => Ok(DataType::Json),
        sp::DataType::Uuid => Ok(DataType::Uuid),
        sp::DataType::Array(elem) => match elem {
            sp::ArrayElemTypeDef::AngleBracket(inner)
            | sp::ArrayElemTypeDef::SquareBracket(inner, _) => {
                Ok(DataType::Array(Box::new(convert_data_type(*inner)?)))
            }
            sp::ArrayElemTypeDef::None => Err(ParseError::Unsupported("untyped array".to_string())),
            sp::ArrayElemTypeDef::Parenthesis(_) => {
                Err(ParseError::Unsupported("parenthesized array type".to_string()))
            }
        },
        sp::DataType::Custom(name, _) => {
            let name_str = name
                .0
                .iter()
                .filter_map(|p| p.as_ident().map(|i| i.value.clone()))
                .collect::<Vec<_>>()
                .join(".");

            // Check for VECTOR type
            if name_str.eq_ignore_ascii_case("vector") {
                Ok(DataType::Vector(None))
            } else {
                Ok(DataType::Custom(name_str))
            }
        }
        _ => Err(ParseError::Unsupported(format!("data type: {dt:?}"))),
    }
}

/// Formats a data type as a string.
fn format_data_type(dt: &sp::DataType) -> String {
    format!("{dt}")
}

/// Converts an object name.
fn convert_object_name(name: sp::ObjectName) -> QualifiedName {
    QualifiedName::new(
        name.0.into_iter().filter_map(object_name_part_to_ident).map(convert_ident).collect(),
    )
}

/// Converts an identifier.
fn convert_ident(ident: sp::Ident) -> Identifier {
    Identifier { name: ident.value, quote_style: ident.quote_style }
}

// ============================================================================
// Transaction Statement Conversions
// ============================================================================

/// Converts a START TRANSACTION or BEGIN statement.
fn convert_start_transaction(
    modes: Vec<sp::TransactionMode>,
    begin: bool,
) -> ParseResult<TransactionStatement> {
    let mut isolation_level = None;
    let mut access_mode = None;

    for mode in modes {
        match mode {
            sp::TransactionMode::IsolationLevel(level) => {
                isolation_level = Some(convert_isolation_level(level)?);
            }
            sp::TransactionMode::AccessMode(mode) => {
                access_mode = Some(convert_access_mode(mode));
            }
        }
    }

    let begin_stmt = BeginTransaction {
        has_transaction_keyword: !begin, // BEGIN doesn't have TRANSACTION keyword by default
        isolation_level,
        access_mode,
        deferred: false,
    };

    Ok(TransactionStatement::Begin(begin_stmt))
}

/// Converts a ROLLBACK statement.
fn convert_rollback(savepoint: Option<sp::Ident>) -> ParseResult<TransactionStatement> {
    let rollback = RollbackTransaction {
        has_transaction_keyword: false,
        to_savepoint: savepoint.map(|s| Identifier::new(s.value)),
    };
    Ok(TransactionStatement::Rollback(rollback))
}

/// Converts a SET TRANSACTION statement.
fn convert_set_transaction(
    modes: Vec<sp::TransactionMode>,
    _snapshot: Option<sp::Value>,
    _session: bool,
) -> ParseResult<TransactionStatement> {
    let mut isolation_level = None;
    let mut access_mode = None;

    for mode in modes {
        match mode {
            sp::TransactionMode::IsolationLevel(level) => {
                isolation_level = Some(convert_isolation_level(level)?);
            }
            sp::TransactionMode::AccessMode(mode) => {
                access_mode = Some(convert_access_mode(mode));
            }
        }
    }

    let set_stmt = SetTransactionStatement { isolation_level, access_mode };

    Ok(TransactionStatement::SetTransaction(set_stmt))
}

/// Converts an isolation level.
fn convert_isolation_level(level: sp::TransactionIsolationLevel) -> ParseResult<IsolationLevel> {
    match level {
        sp::TransactionIsolationLevel::ReadUncommitted => Ok(IsolationLevel::ReadUncommitted),
        sp::TransactionIsolationLevel::ReadCommitted => Ok(IsolationLevel::ReadCommitted),
        sp::TransactionIsolationLevel::RepeatableRead => Ok(IsolationLevel::RepeatableRead),
        sp::TransactionIsolationLevel::Serializable => Ok(IsolationLevel::Serializable),
        sp::TransactionIsolationLevel::Snapshot => {
            Err(ParseError::Unsupported("SNAPSHOT isolation level is not supported".to_string()))
        }
    }
}

/// Converts an access mode.
fn convert_access_mode(mode: sp::TransactionAccessMode) -> TransactionAccessMode {
    match mode {
        sp::TransactionAccessMode::ReadOnly => TransactionAccessMode::ReadOnly,
        sp::TransactionAccessMode::ReadWrite => TransactionAccessMode::ReadWrite,
    }
}

// ============================================================================
// Utility Statement Conversions
// ============================================================================

/// Converts EXPLAIN ANALYZE options to our statement.
fn convert_explain_analyze(
    statement: Statement,
    verbose: bool,
    format: Option<sp::AnalyzeFormatKind>,
) -> ParseResult<ExplainAnalyzeStatement> {
    // Extract the AnalyzeFormat from AnalyzeFormatKind wrapper
    let analyze_format = format.map(|kind| match kind {
        sp::AnalyzeFormatKind::Keyword(f) | sp::AnalyzeFormatKind::Assignment(f) => f,
    });
    let explain_format = match analyze_format {
        Some(sp::AnalyzeFormat::TEXT) | None => ExplainFormat::Text,
        Some(sp::AnalyzeFormat::JSON) => ExplainFormat::Json,
        Some(sp::AnalyzeFormat::GRAPHVIZ) => ExplainFormat::Text, // Fall back to text
        Some(sp::AnalyzeFormat::TRADITIONAL) => ExplainFormat::Text, // Fall back to text
        Some(sp::AnalyzeFormat::TREE) => ExplainFormat::Text,     // Fall back to text
    };

    Ok(ExplainAnalyzeStatement {
        statement,
        buffers: false, // Not supported by default
        timing: true,   // Default true
        format: explain_format,
        verbose,
        costs: true,     // Default true
        settings: false, // Not supported
    })
}

/// Converts a COPY statement.
#[allow(clippy::too_many_arguments)]
fn convert_copy(
    source: sp::CopySource,
    to: bool,
    target: sp::CopyTarget,
    options: Vec<sp::CopyOption>,
    _values: Vec<Option<String>>,
) -> ParseResult<CopyStatement> {
    // Convert source (table or query)
    let copy_target = match source {
        sp::CopySource::Table { table_name, columns } => CopyTarget::Table {
            name: convert_object_name(table_name),
            columns: columns.into_iter().map(convert_ident).collect(),
        },
        sp::CopySource::Query(query) => CopyTarget::Query(Box::new(convert_query(*query)?)),
    };

    // Convert target (file path or stdout/stdin)
    let direction = if to {
        // COPY TO
        let dest = match target {
            sp::CopyTarget::File { filename } => CopyDestination::File(filename),
            sp::CopyTarget::Stdout => CopyDestination::Stdout,
            sp::CopyTarget::Program { command } => CopyDestination::Program(command),
            sp::CopyTarget::Stdin => {
                return Err(ParseError::SqlSyntax("cannot use STDIN with COPY TO".to_string()));
            }
        };
        CopyDirection::To(dest)
    } else {
        // COPY FROM
        let src = match target {
            sp::CopyTarget::File { filename } => CopySource::File(filename),
            sp::CopyTarget::Stdin => CopySource::Stdin,
            sp::CopyTarget::Program { command } => CopySource::Program(command),
            sp::CopyTarget::Stdout => {
                return Err(ParseError::SqlSyntax("cannot use STDOUT with COPY FROM".to_string()));
            }
        };
        CopyDirection::From(src)
    };

    // Convert options
    let mut copy_options = CopyOptions::default();
    for opt in options {
        match opt {
            sp::CopyOption::Format(f) => {
                copy_options.format = match f.value.to_uppercase().as_str() {
                    "CSV" => CopyFormat::Csv,
                    "TEXT" => CopyFormat::Text,
                    "BINARY" => CopyFormat::Binary,
                    _ => CopyFormat::Text,
                };
            }
            sp::CopyOption::Header(h) => copy_options.header = h,
            sp::CopyOption::Delimiter(d) => {
                copy_options.delimiter = Some(d);
            }
            sp::CopyOption::Null(n) => copy_options.null_string = Some(n),
            sp::CopyOption::Quote(q) => copy_options.quote = Some(q),
            sp::CopyOption::Escape(e) => copy_options.escape = Some(e),
            sp::CopyOption::Encoding(e) => copy_options.encoding = Some(e),
            sp::CopyOption::ForceQuote(cols) => {
                copy_options.force_quote = cols.into_iter().map(convert_ident).collect();
            }
            sp::CopyOption::ForceNotNull(cols) => {
                copy_options.force_not_null = cols.into_iter().map(convert_ident).collect();
            }
            // Ignore other options
            _ => {}
        }
    }

    Ok(CopyStatement { target: copy_target, direction, options: copy_options })
}

/// Converts a SET SingleAssignment statement (new API in sqlparser 0.60).
fn convert_set_single_assignment(
    variable: sp::ObjectName,
    values: Vec<sp::Expr>,
    local: bool,
) -> ParseResult<SetSessionStatement> {
    // Get the variable name from the first part of the ObjectName
    let name = variable
        .0
        .into_iter()
        .next()
        .and_then(object_name_part_to_ident)
        .map(convert_ident)
        .ok_or_else(|| ParseError::MissingClause("variable name in SET statement".to_string()))?;

    // Convert values
    let set_value = if values.is_empty() {
        None // SET x TO DEFAULT
    } else {
        // Consume the vector and get an iterator
        let mut iter = values.into_iter();
        let first_expr = iter
            .next()
            .ok_or_else(|| ParseError::MissingClause("value in SET statement".to_string()))?;

        // Check if there are more values (multi-value list)
        let remaining: Vec<_> = iter.collect();
        if remaining.is_empty() {
            // Single value
            Some(SetValue::Single(convert_expr(first_expr)?))
        } else {
            // Multiple values - collect all including the first
            let mut all_values = vec![convert_expr(first_expr)?];
            for expr in remaining {
                all_values.push(convert_expr(expr)?);
            }
            Some(SetValue::List(all_values))
        }
    };

    Ok(SetSessionStatement { name, value: set_value, local })
}

/// Converts a SHOW statement.
fn convert_show_variable(variable: Vec<sp::Ident>) -> ParseResult<ShowStatement> {
    let name = if variable.is_empty()
        || (variable.len() == 1 && variable[0].value.eq_ignore_ascii_case("ALL"))
    {
        None // SHOW ALL
    } else {
        // Join multi-part variable names
        let name_str = variable.iter().map(|i| i.value.clone()).collect::<Vec<_>>().join(".");
        Some(Identifier::new(name_str))
    };

    Ok(ShowStatement { name })
}

/// Converts an ANALYZE statement.
fn convert_analyze(
    table_name: sp::ObjectName,
    _partitions: Option<Vec<sp::Expr>>,
    columns: Vec<sp::Ident>,
) -> ParseResult<AnalyzeStatement> {
    let table = Some(convert_object_name(table_name));
    let columns = columns.into_iter().map(convert_ident).collect();

    Ok(AnalyzeStatement { table, columns })
}

/// Converts a VACUUM statement.
fn convert_vacuum(vacuum: sp::VacuumStatement) -> ParseResult<VacuumStatement> {
    let table = vacuum.table_name.map(convert_object_name);

    // sqlparser's VacuumStatement doesn't have analyze or columns fields,
    // but it does have full. Our VacuumStatement supports analyze for VACUUM ANALYZE.
    Ok(VacuumStatement {
        full: vacuum.full,
        analyze: false, // Standard VACUUM doesn't combine with ANALYZE in sqlparser
        table,
        columns: vec![],
    })
}

/// Converts a RESET statement.
fn convert_reset(reset: sp::ResetStatement) -> ParseResult<ResetStatement> {
    let name = match reset.reset {
        sp::Reset::ALL => None,
        sp::Reset::ConfigurationParameter(name) => {
            // Join multi-part names
            let name_str = name.0.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(".");
            Some(Identifier::new(name_str))
        }
    };

    Ok(ResetStatement { name })
}

/// Converts a CREATE SCHEMA statement.
fn convert_create_schema(
    schema_name: sp::SchemaName,
    if_not_exists: bool,
) -> ParseResult<CreateSchemaStatement> {
    // Extract both name and optional authorization from schema_name
    let (name, authorization) = match schema_name {
        sp::SchemaName::Simple(name) => (Identifier::new(name.to_string()), None),
        sp::SchemaName::UnnamedAuthorization(ident) => {
            // Only authorization, no name - use authorization as name
            let auth = Identifier::new(ident.to_string());
            (auth.clone(), Some(auth))
        }
        sp::SchemaName::NamedAuthorization(name, auth) => {
            (Identifier::new(name.to_string()), Some(Identifier::new(auth.to_string())))
        }
    };

    Ok(CreateSchemaStatement { if_not_exists, name, authorization })
}

/// Converts a CREATE FUNCTION statement.
fn convert_create_function(
    create_func: sp::CreateFunction,
) -> ParseResult<CreateFunctionStatement> {
    // Convert parameters
    let parameters = create_func
        .args
        .unwrap_or_default()
        .into_iter()
        .map(|arg| {
            let mode = match arg.mode {
                Some(sp::ArgMode::In) => ParameterMode::In,
                Some(sp::ArgMode::Out) => ParameterMode::Out,
                Some(sp::ArgMode::InOut) => ParameterMode::InOut,
                None => ParameterMode::In,
            };
            let name = arg.name.map(|n| Identifier::new(n.to_string()));
            // data_type is required DataType, convert with fallback to Custom for unsupported types
            let data_type = convert_data_type(arg.data_type.clone())
                .unwrap_or_else(|_| DataType::Custom(arg.data_type.to_string()));
            let default = arg.default_expr.map(|e| e.to_string());

            FunctionParameter { name, mode, data_type, default }
        })
        .collect();

    // Convert return type (default to VARCHAR if not specified)
    let returns = create_func
        .return_type
        .and_then(|rt| convert_data_type(rt).ok())
        .unwrap_or(DataType::Varchar(None));

    // Convert function body - extract the body expression from CreateFunctionBody enum
    let body = create_func
        .function_body
        .map(|fb| match fb {
            sp::CreateFunctionBody::AsBeforeOptions { body, .. } => body.to_string(),
            sp::CreateFunctionBody::AsAfterOptions(expr) => expr.to_string(),
            sp::CreateFunctionBody::Return(expr) => expr.to_string(),
            sp::CreateFunctionBody::AsReturnExpr(expr) => expr.to_string(),
            sp::CreateFunctionBody::AsReturnSelect(query) => query.to_string(),
            sp::CreateFunctionBody::AsBeginEnd(bes) => bes.to_string(),
        })
        .unwrap_or_default();

    // Convert language (default to SQL)
    let func_language = create_func
        .language
        .map(|l| {
            let lang_str = l.to_string().to_uppercase();
            match lang_str.as_str() {
                "SQL" => FunctionLanguage::Sql,
                "PLPGSQL" => FunctionLanguage::PlPgSql,
                _ => FunctionLanguage::Sql,
            }
        })
        .unwrap_or(FunctionLanguage::Sql);

    // Convert volatility
    let volatility = create_func.behavior.map(|b| match b {
        sp::FunctionBehavior::Immutable => FunctionVolatility::Immutable,
        sp::FunctionBehavior::Stable => FunctionVolatility::Stable,
        sp::FunctionBehavior::Volatile => FunctionVolatility::Volatile,
    });

    Ok(CreateFunctionStatement {
        or_replace: create_func.or_replace,
        name: QualifiedName::simple(create_func.name.to_string()),
        parameters,
        returns,
        returns_setof: false,
        body,
        language: func_language,
        volatility,
        strict: false,
        security_definer: false,
    })
}

/// Converts a DROP FUNCTION statement.
fn convert_drop_function(drop_func: sp::DropFunction) -> ParseResult<DropFunctionStatement> {
    // Get the first function name (most common case is a single function)
    let name = drop_func
        .func_desc
        .first()
        .map(|fd| QualifiedName::simple(fd.name.to_string()))
        .ok_or_else(|| {
            ParseError::SqlSyntax("DROP FUNCTION requires a function name".to_string())
        })?;

    // Convert argument types for overload resolution
    // OperateFunctionArg.data_type is DataType (not Option), so convert directly
    let arg_types = drop_func
        .func_desc
        .first()
        .and_then(|fd| fd.args.as_ref())
        .map(|args| {
            args.iter().filter_map(|arg| convert_data_type(arg.data_type.clone()).ok()).collect()
        })
        .unwrap_or_default();

    let cascade = matches!(drop_func.drop_behavior, Some(sp::DropBehavior::Cascade));

    Ok(DropFunctionStatement { if_exists: drop_func.if_exists, name, arg_types, cascade })
}

/// Converts a CREATE TRIGGER statement.
fn convert_create_trigger(
    create_trigger: sp::CreateTrigger,
) -> ParseResult<CreateTriggerStatement> {
    // Convert timing - For is not a standard timing, default to Before in that case
    let timing = match create_trigger.period {
        Some(sp::TriggerPeriod::Before) => TriggerTiming::Before,
        Some(sp::TriggerPeriod::After) => TriggerTiming::After,
        Some(sp::TriggerPeriod::InsteadOf) => TriggerTiming::InsteadOf,
        Some(sp::TriggerPeriod::For) | None => TriggerTiming::Before, // Default to BEFORE
    };

    // Convert events
    let trigger_events = create_trigger
        .events
        .into_iter()
        .map(|e| match e {
            sp::TriggerEvent::Insert => TriggerEvent::Insert,
            sp::TriggerEvent::Update(cols) => {
                let columns = cols.into_iter().map(|c| Identifier::new(c.to_string())).collect();
                TriggerEvent::Update(columns)
            }
            sp::TriggerEvent::Delete => TriggerEvent::Delete,
            sp::TriggerEvent::Truncate => TriggerEvent::Truncate,
        })
        .collect();

    // Get table name as QualifiedName
    let table = QualifiedName::simple(create_trigger.table_name.to_string());

    // Extract function from exec_body - func_desc contains the function name
    let function = create_trigger
        .exec_body
        .map(|eb| QualifiedName::simple(eb.func_desc.name.to_string()))
        .unwrap_or_else(|| QualifiedName::simple(String::new()));

    Ok(CreateTriggerStatement {
        or_replace: create_trigger.or_replace,
        name: Identifier::new(create_trigger.name.to_string()),
        timing,
        events: trigger_events,
        table,
        for_each: TriggerForEach::Row, // Default to ROW
        when_clause: None,
        function,
        function_args: vec![],
    })
}

/// Converts a DROP TRIGGER statement.
fn convert_drop_trigger(drop_trigger: sp::DropTrigger) -> ParseResult<DropTriggerStatement> {
    let cascade = matches!(drop_trigger.option, Some(sp::ReferentialAction::Cascade));

    // table_name is Option<ObjectName>, convert to QualifiedName
    let table = drop_trigger
        .table_name
        .map(|t| QualifiedName::simple(t.to_string()))
        .unwrap_or_else(|| QualifiedName::simple(String::new()));

    Ok(DropTriggerStatement {
        if_exists: drop_trigger.if_exists,
        name: Identifier::new(drop_trigger.trigger_name.to_string()),
        table,
        cascade,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_select() {
        let stmt = parse_single_statement("SELECT * FROM users").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 1);
                assert!(matches!(select.projection[0], SelectItem::Wildcard));
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_with_where() {
        let stmt = parse_single_statement("SELECT id, name FROM users WHERE id = 1").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 2);
                assert!(select.where_clause.is_some());
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_insert() {
        let stmt =
            parse_single_statement("INSERT INTO users (name, age) VALUES ('Alice', 30)").unwrap();
        match stmt {
            Statement::Insert(insert) => {
                assert_eq!(insert.columns.len(), 2);
                match &insert.source {
                    InsertSource::Values(rows) => {
                        assert_eq!(rows.len(), 1);
                        assert_eq!(rows[0].len(), 2);
                    }
                    _ => panic!("expected VALUES"),
                }
            }
            _ => panic!("expected INSERT"),
        }
    }

    #[test]
    fn parse_update() {
        let stmt = parse_single_statement("UPDATE users SET name = 'Bob' WHERE id = 1").unwrap();
        match stmt {
            Statement::Update(update) => {
                assert_eq!(update.assignments.len(), 1);
                assert!(update.where_clause.is_some());
            }
            _ => panic!("expected UPDATE"),
        }
    }

    #[test]
    fn parse_delete() {
        let stmt = parse_single_statement("DELETE FROM users WHERE id = 1").unwrap();
        match stmt {
            Statement::Delete(delete) => {
                assert!(delete.where_clause.is_some());
            }
            _ => panic!("expected DELETE"),
        }
    }

    #[test]
    fn parse_create_table() {
        let stmt = parse_single_statement(
            "CREATE TABLE users (id BIGINT PRIMARY KEY, name VARCHAR(100) NOT NULL)",
        )
        .unwrap();
        match stmt {
            Statement::CreateTable(create) => {
                assert_eq!(create.columns.len(), 2);
            }
            _ => panic!("expected CREATE TABLE"),
        }
    }

    #[test]
    fn parse_join() {
        let stmt = parse_single_statement(
            "SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.from.len(), 1);
                match &select.from[0] {
                    TableRef::Join(join) => {
                        assert_eq!(join.join_type, JoinType::Inner);
                    }
                    _ => panic!("expected JOIN"),
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_empty_query() {
        let result = parse_sql("");
        assert!(matches!(result, Err(ParseError::EmptyQuery)));
    }

    #[test]
    fn parse_parameter() {
        let stmt = parse_single_statement("SELECT * FROM users WHERE id = $1").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::BinaryOp { right, .. }) = select.where_clause {
                    match *right {
                        Expr::Parameter(ParameterRef::Positional(1)) => {}
                        _ => panic!("expected positional parameter"),
                    }
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_vector_literal() {
        let stmt = parse_single_statement("SELECT [0.1, 0.2, 0.3]").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 1);
                if let SelectItem::Expr { expr, .. } = &select.projection[0] {
                    match expr {
                        Expr::Literal(Literal::Vector(v)) => {
                            assert_eq!(v.len(), 3);
                            assert!((v[0] - 0.1).abs() < 0.001);
                            assert!((v[1] - 0.2).abs() < 0.001);
                            assert!((v[2] - 0.3).abs() < 0.001);
                        }
                        _ => panic!("expected Vector literal, got {:?}", expr),
                    }
                } else {
                    panic!("expected expression in projection");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multi_vector_literal() {
        let stmt = parse_single_statement("SELECT [[0.1, 0.2], [0.3, 0.4]]").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 1);
                if let SelectItem::Expr { expr, .. } = &select.projection[0] {
                    match expr {
                        Expr::Literal(Literal::MultiVector(v)) => {
                            assert_eq!(v.len(), 2);
                            assert_eq!(v[0].len(), 2);
                            assert_eq!(v[1].len(), 2);
                            assert!((v[0][0] - 0.1).abs() < 0.001);
                            assert!((v[0][1] - 0.2).abs() < 0.001);
                            assert!((v[1][0] - 0.3).abs() < 0.001);
                            assert!((v[1][1] - 0.4).abs() < 0.001);
                        }
                        _ => panic!("expected MultiVector literal, got {:?}", expr),
                    }
                } else {
                    panic!("expected expression in projection");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multi_vector_in_order_by() {
        // This tests that multi-vector literals can appear in ORDER BY clauses
        // The actual operator <##> will be handled by the extensions parser
        let stmt = parse_single_statement(
            "SELECT * FROM docs ORDER BY embedding <-> [[0.1, 0.2], [0.3, 0.4]]",
        );
        // This will fail parsing due to <-> which needs the extensions parser
        // But we're just testing the multi-vector parsing capability here
        assert!(stmt.is_err()); // <-> is not a standard SQL operator
    }

    #[test]
    fn parse_insert_with_multi_vector() {
        let stmt = parse_single_statement(
            "INSERT INTO docs (id, embedding) VALUES (1, [[0.1, 0.2], [0.3, 0.4]])",
        )
        .unwrap();
        match stmt {
            Statement::Insert(insert) => {
                assert_eq!(insert.columns.len(), 2);
                match &insert.source {
                    InsertSource::Values(rows) => {
                        assert_eq!(rows.len(), 1);
                        assert_eq!(rows[0].len(), 2);
                        match &rows[0][1] {
                            Expr::Literal(Literal::MultiVector(v)) => {
                                assert_eq!(v.len(), 2);
                                assert_eq!(v[0].len(), 2);
                            }
                            _ => panic!("expected MultiVector literal in insert"),
                        }
                    }
                    _ => panic!("expected VALUES"),
                }
            }
            _ => panic!("expected INSERT"),
        }
    }

    // =====================
    // VIEW Tests
    // =====================

    #[test]
    fn parse_create_view_basic() {
        let stmt = parse_single_statement(
            "CREATE VIEW active_users AS SELECT * FROM users WHERE status = 'active'",
        )
        .unwrap();
        match stmt {
            Statement::CreateView(view) => {
                assert_eq!(view.name.to_string(), "active_users");
                assert!(!view.or_replace);
                assert!(view.columns.is_empty());
            }
            _ => panic!("expected CREATE VIEW"),
        }
    }

    #[test]
    fn parse_create_or_replace_view() {
        let stmt = parse_single_statement(
            "CREATE OR REPLACE VIEW user_stats AS SELECT department, COUNT(*) as count FROM employees GROUP BY department",
        )
        .unwrap();
        match stmt {
            Statement::CreateView(view) => {
                assert_eq!(view.name.to_string(), "user_stats");
                assert!(view.or_replace);
                assert!(view.columns.is_empty());
            }
            _ => panic!("expected CREATE VIEW"),
        }
    }

    #[test]
    fn parse_create_view_with_columns() {
        let stmt = parse_single_statement("CREATE VIEW my_view (col1, col2) AS SELECT a, b FROM t")
            .unwrap();
        match stmt {
            Statement::CreateView(view) => {
                assert_eq!(view.name.to_string(), "my_view");
                assert_eq!(view.columns.len(), 2);
                assert_eq!(view.columns[0].name, "col1");
                assert_eq!(view.columns[1].name, "col2");
            }
            _ => panic!("expected CREATE VIEW"),
        }
    }

    #[test]
    fn parse_create_view_with_join() {
        let stmt = parse_single_statement(
            "CREATE VIEW order_summary AS SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
        )
        .unwrap();
        match stmt {
            Statement::CreateView(view) => {
                assert_eq!(view.name.to_string(), "order_summary");
                // The query should contain the join
                assert!(view.query.from.len() == 1);
            }
            _ => panic!("expected CREATE VIEW"),
        }
    }

    #[test]
    fn parse_drop_view_basic() {
        let stmt = parse_single_statement("DROP VIEW active_users").unwrap();
        match stmt {
            Statement::DropView(drop) => {
                assert!(!drop.if_exists);
                assert_eq!(drop.names.len(), 1);
                assert_eq!(drop.names[0].to_string(), "active_users");
                assert!(!drop.cascade);
            }
            _ => panic!("expected DROP VIEW"),
        }
    }

    #[test]
    fn parse_drop_view_if_exists() {
        let stmt = parse_single_statement("DROP VIEW IF EXISTS maybe_exists").unwrap();
        match stmt {
            Statement::DropView(drop) => {
                assert!(drop.if_exists);
                assert_eq!(drop.names.len(), 1);
                assert_eq!(drop.names[0].to_string(), "maybe_exists");
            }
            _ => panic!("expected DROP VIEW"),
        }
    }

    #[test]
    fn parse_drop_view_cascade() {
        let stmt = parse_single_statement("DROP VIEW cascade_view CASCADE").unwrap();
        match stmt {
            Statement::DropView(drop) => {
                assert!(!drop.if_exists);
                assert!(drop.cascade);
                assert_eq!(drop.names.len(), 1);
            }
            _ => panic!("expected DROP VIEW"),
        }
    }

    #[test]
    fn parse_drop_view_multiple() {
        let stmt = parse_single_statement("DROP VIEW view1, view2, view3").unwrap();
        match stmt {
            Statement::DropView(drop) => {
                assert_eq!(drop.names.len(), 3);
                assert_eq!(drop.names[0].to_string(), "view1");
                assert_eq!(drop.names[1].to_string(), "view2");
                assert_eq!(drop.names[2].to_string(), "view3");
            }
            _ => panic!("expected DROP VIEW"),
        }
    }

    #[test]
    fn parse_drop_view_if_exists_cascade() {
        let stmt = parse_single_statement("DROP VIEW IF EXISTS my_view CASCADE").unwrap();
        match stmt {
            Statement::DropView(drop) => {
                assert!(drop.if_exists);
                assert!(drop.cascade);
            }
            _ => panic!("expected DROP VIEW"),
        }
    }

    // =====================
    // ALTER TABLE Tests
    // =====================

    #[test]
    fn parse_alter_table_add_column() {
        let stmt =
            parse_single_statement("ALTER TABLE users ADD COLUMN email VARCHAR(255)").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.name.to_string(), "users");
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::AddColumn { if_not_exists, column } => {
                        assert!(!if_not_exists);
                        assert_eq!(column.name.name, "email");
                    }
                    _ => panic!("expected ADD COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_drop_column() {
        let stmt = parse_single_statement("ALTER TABLE users DROP COLUMN temp_field").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.name.to_string(), "users");
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::DropColumn { if_exists, column_name, cascade } => {
                        assert!(!if_exists);
                        assert!(!cascade);
                        assert_eq!(column_name.name, "temp_field");
                    }
                    _ => panic!("expected DROP COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_drop_column_if_exists() {
        let stmt =
            parse_single_statement("ALTER TABLE users DROP COLUMN IF EXISTS maybe_exists").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::DropColumn { if_exists, column_name, .. } => {
                        assert!(*if_exists);
                        assert_eq!(column_name.name, "maybe_exists");
                    }
                    _ => panic!("expected DROP COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_alter_column_set_not_null() {
        let stmt =
            parse_single_statement("ALTER TABLE users ALTER COLUMN name SET NOT NULL").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::AlterColumn { column_name, action } => {
                        assert_eq!(column_name.name, "name");
                        assert!(matches!(action, AlterColumnAction::SetNotNull));
                    }
                    _ => panic!("expected ALTER COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_alter_column_set_default() {
        let stmt =
            parse_single_statement("ALTER TABLE users ALTER COLUMN age SET DEFAULT 0").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::AlterColumn { column_name, action } => {
                        assert_eq!(column_name.name, "age");
                        match action {
                            AlterColumnAction::SetDefault(expr) => match expr {
                                Expr::Literal(Literal::Integer(0)) => {}
                                _ => panic!("expected integer literal 0"),
                            },
                            _ => panic!("expected SET DEFAULT"),
                        }
                    }
                    _ => panic!("expected ALTER COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_alter_column_type() {
        let stmt = parse_single_statement(
            "ALTER TABLE users ALTER COLUMN score SET DATA TYPE DOUBLE PRECISION",
        )
        .unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::AlterColumn { column_name, action } => {
                        assert_eq!(column_name.name, "score");
                        match action {
                            AlterColumnAction::SetType { data_type, using } => {
                                assert!(matches!(data_type, DataType::DoublePrecision));
                                assert!(using.is_none());
                            }
                            _ => panic!("expected SET TYPE"),
                        }
                    }
                    _ => panic!("expected ALTER COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_rename_column() {
        let stmt =
            parse_single_statement("ALTER TABLE users RENAME COLUMN old_name TO new_name").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::RenameColumn { old_name, new_name } => {
                        assert_eq!(old_name.name, "old_name");
                        assert_eq!(new_name.name, "new_name");
                    }
                    _ => panic!("expected RENAME COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_rename_table() {
        let stmt = parse_single_statement("ALTER TABLE old_table RENAME TO new_table").unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.name.to_string(), "old_table");
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::RenameTable { new_name } => {
                        assert_eq!(new_name.to_string(), "new_table");
                    }
                    _ => panic!("expected RENAME TO"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    #[test]
    fn parse_alter_table_add_column_with_default() {
        let stmt = parse_single_statement(
            "ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT now()",
        )
        .unwrap();
        match stmt {
            Statement::AlterTable(alter) => {
                assert_eq!(alter.actions.len(), 1);
                match &alter.actions[0] {
                    AlterTableAction::AddColumn { column, .. } => {
                        assert_eq!(column.name.name, "created_at");
                        assert!(matches!(column.data_type, DataType::Timestamp));
                        // Check for default constraint
                        assert!(column
                            .constraints
                            .iter()
                            .any(|c| matches!(c, ColumnConstraint::Default(_))));
                    }
                    _ => panic!("expected ADD COLUMN"),
                }
            }
            _ => panic!("expected ALTER TABLE"),
        }
    }

    // ========================================================================
    // Transaction Statement Tests
    // ========================================================================

    #[test]
    fn parse_begin() {
        let stmt = parse_single_statement("BEGIN").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Begin(begin)) => {
                assert!(begin.isolation_level.is_none());
                assert!(begin.access_mode.is_none());
            }
            _ => panic!("expected BEGIN"),
        }
    }

    #[test]
    fn parse_start_transaction() {
        let stmt = parse_single_statement("START TRANSACTION").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Begin(begin)) => {
                assert!(begin.isolation_level.is_none());
                assert!(begin.access_mode.is_none());
            }
            _ => panic!("expected START TRANSACTION"),
        }
    }

    #[test]
    fn parse_begin_with_isolation_level() {
        let stmt =
            parse_single_statement("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Begin(begin)) => {
                assert_eq!(begin.isolation_level, Some(IsolationLevel::Serializable));
                assert!(begin.access_mode.is_none());
            }
            _ => panic!("expected BEGIN with isolation level"),
        }
    }

    #[test]
    fn parse_begin_read_only() {
        let stmt = parse_single_statement("START TRANSACTION READ ONLY").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Begin(begin)) => {
                assert_eq!(begin.access_mode, Some(TransactionAccessMode::ReadOnly));
            }
            _ => panic!("expected START TRANSACTION READ ONLY"),
        }
    }

    #[test]
    fn parse_begin_repeatable_read() {
        let stmt =
            parse_single_statement("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Begin(begin)) => {
                assert_eq!(begin.isolation_level, Some(IsolationLevel::RepeatableRead));
            }
            _ => panic!("expected BEGIN with REPEATABLE READ"),
        }
    }

    #[test]
    fn parse_commit() {
        let stmt = parse_single_statement("COMMIT").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Commit) => {}
            _ => panic!("expected COMMIT"),
        }
    }

    #[test]
    fn parse_rollback() {
        let stmt = parse_single_statement("ROLLBACK").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Rollback(rollback)) => {
                assert!(rollback.to_savepoint.is_none());
            }
            _ => panic!("expected ROLLBACK"),
        }
    }

    #[test]
    fn parse_rollback_to_savepoint() {
        let stmt = parse_single_statement("ROLLBACK TO SAVEPOINT sp1").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Rollback(rollback)) => {
                assert_eq!(rollback.to_savepoint.as_ref().map(|s| s.name.as_str()), Some("sp1"));
            }
            _ => panic!("expected ROLLBACK TO SAVEPOINT"),
        }
    }

    #[test]
    fn parse_savepoint() {
        let stmt = parse_single_statement("SAVEPOINT my_savepoint").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::Savepoint(sp)) => {
                assert_eq!(sp.name.name, "my_savepoint");
            }
            _ => panic!("expected SAVEPOINT"),
        }
    }

    #[test]
    fn parse_release_savepoint() {
        let stmt = parse_single_statement("RELEASE SAVEPOINT my_savepoint").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::ReleaseSavepoint(release)) => {
                assert_eq!(release.name.name, "my_savepoint");
            }
            _ => panic!("expected RELEASE SAVEPOINT"),
        }
    }

    #[test]
    fn parse_set_transaction() {
        let stmt =
            parse_single_statement("SET TRANSACTION ISOLATION LEVEL READ COMMITTED").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::SetTransaction(set_txn)) => {
                assert_eq!(set_txn.isolation_level, Some(IsolationLevel::ReadCommitted));
            }
            _ => panic!("expected SET TRANSACTION"),
        }
    }

    #[test]
    fn parse_set_transaction_read_write() {
        let stmt = parse_single_statement("SET TRANSACTION READ WRITE").unwrap();
        match stmt {
            Statement::Transaction(TransactionStatement::SetTransaction(set_txn)) => {
                assert_eq!(set_txn.access_mode, Some(TransactionAccessMode::ReadWrite));
            }
            _ => panic!("expected SET TRANSACTION READ WRITE"),
        }
    }

    #[test]
    fn parse_transaction_sequence() {
        // Test parsing multiple transaction statements in sequence
        let stmts = parse_sql(
            "BEGIN; INSERT INTO users VALUES (1, 'Alice'); SAVEPOINT sp1; ROLLBACK TO SAVEPOINT sp1; COMMIT"
        ).unwrap();
        assert_eq!(stmts.len(), 5);

        assert!(matches!(&stmts[0], Statement::Transaction(TransactionStatement::Begin(_))));
        assert!(matches!(&stmts[1], Statement::Insert(_)));
        assert!(matches!(&stmts[2], Statement::Transaction(TransactionStatement::Savepoint(_))));
        assert!(matches!(&stmts[3], Statement::Transaction(TransactionStatement::Rollback(_))));
        assert!(matches!(&stmts[4], Statement::Transaction(TransactionStatement::Commit)));
    }

    // =====================
    // VACUUM Tests
    // =====================

    #[test]
    fn parse_vacuum() {
        let stmt = parse_single_statement("VACUUM").unwrap();
        match stmt {
            Statement::Utility(utility) => match *utility {
                UtilityStatement::Vacuum(vacuum) => {
                    assert!(!vacuum.full);
                    assert!(!vacuum.analyze);
                    assert!(vacuum.table.is_none());
                    assert!(vacuum.columns.is_empty());
                }
                _ => panic!("expected VACUUM"),
            },
            _ => panic!("expected Utility statement"),
        }
    }

    #[test]
    fn parse_vacuum_table() {
        let stmt = parse_single_statement("VACUUM users").unwrap();
        match stmt {
            Statement::Utility(utility) => match *utility {
                UtilityStatement::Vacuum(vacuum) => {
                    assert!(!vacuum.full);
                    assert!(!vacuum.analyze);
                    assert_eq!(
                        vacuum.table.as_ref().map(|t| t.to_string()),
                        Some("users".to_string())
                    );
                }
                _ => panic!("expected VACUUM"),
            },
            _ => panic!("expected Utility statement"),
        }
    }

    #[test]
    fn parse_vacuum_full() {
        let stmt = parse_single_statement("VACUUM FULL users").unwrap();
        match stmt {
            Statement::Utility(utility) => match *utility {
                UtilityStatement::Vacuum(vacuum) => {
                    assert!(vacuum.full);
                    assert!(!vacuum.analyze);
                    assert_eq!(
                        vacuum.table.as_ref().map(|t| t.to_string()),
                        Some("users".to_string())
                    );
                }
                _ => panic!("expected VACUUM"),
            },
            _ => panic!("expected Utility statement"),
        }
    }

    // =====================
    // RESET Tests
    // =====================

    #[test]
    fn parse_reset_all() {
        let stmt = parse_single_statement("RESET ALL").unwrap();
        match stmt {
            Statement::Utility(utility) => match *utility {
                UtilityStatement::Reset(reset) => {
                    assert!(reset.name.is_none());
                }
                _ => panic!("expected RESET"),
            },
            _ => panic!("expected Utility statement"),
        }
    }

    #[test]
    fn parse_reset_variable() {
        let stmt = parse_single_statement("RESET timezone").unwrap();
        match stmt {
            Statement::Utility(utility) => match *utility {
                UtilityStatement::Reset(reset) => {
                    assert_eq!(reset.name.as_ref().map(|n| n.name.as_str()), Some("timezone"));
                }
                _ => panic!("expected RESET"),
            },
            _ => panic!("expected Utility statement"),
        }
    }

    #[test]
    fn parse_reset_search_path() {
        let stmt = parse_single_statement("RESET search_path").unwrap();
        match stmt {
            Statement::Utility(utility) => match *utility {
                UtilityStatement::Reset(reset) => {
                    assert_eq!(reset.name.as_ref().map(|n| n.name.as_str()), Some("search_path"));
                }
                _ => panic!("expected RESET"),
            },
            _ => panic!("expected Utility statement"),
        }
    }

    // =====================================================
    // Tests for Schema DDL statements
    // =====================================================

    #[test]
    fn parse_create_schema() {
        let stmt = parse_single_statement("CREATE SCHEMA myschema").unwrap();
        match stmt {
            Statement::CreateSchema(create) => {
                assert_eq!(create.name.to_string(), "myschema");
                assert!(!create.if_not_exists);
                assert!(create.authorization.is_none());
            }
            _ => panic!("expected CREATE SCHEMA"),
        }
    }

    #[test]
    fn parse_create_schema_if_not_exists() {
        let stmt = parse_single_statement("CREATE SCHEMA IF NOT EXISTS myschema").unwrap();
        match stmt {
            Statement::CreateSchema(create) => {
                assert_eq!(create.name.to_string(), "myschema");
                assert!(create.if_not_exists);
            }
            _ => panic!("expected CREATE SCHEMA"),
        }
    }

    #[test]
    fn parse_create_schema_authorization() {
        let stmt = parse_single_statement("CREATE SCHEMA myschema AUTHORIZATION myuser").unwrap();
        match stmt {
            Statement::CreateSchema(create) => {
                assert_eq!(create.name.to_string(), "myschema");
                assert!(create.authorization.is_some());
                assert_eq!(create.authorization.unwrap().name, "myuser");
            }
            _ => panic!("expected CREATE SCHEMA"),
        }
    }

    #[test]
    fn parse_drop_schema() {
        let stmt = parse_single_statement("DROP SCHEMA myschema").unwrap();
        match stmt {
            Statement::DropSchema(drop) => {
                assert_eq!(drop.names.len(), 1);
                assert_eq!(drop.names[0].name, "myschema");
                assert!(!drop.if_exists);
                assert!(!drop.cascade);
            }
            _ => panic!("expected DROP SCHEMA"),
        }
    }

    #[test]
    fn parse_drop_schema_if_exists_cascade() {
        let stmt = parse_single_statement("DROP SCHEMA IF EXISTS myschema CASCADE").unwrap();
        match stmt {
            Statement::DropSchema(drop) => {
                assert_eq!(drop.names.len(), 1);
                assert_eq!(drop.names[0].name, "myschema");
                assert!(drop.if_exists);
                assert!(drop.cascade);
            }
            _ => panic!("expected DROP SCHEMA"),
        }
    }

    // =====================================================
    // Tests for Function DDL statements
    // =====================================================

    #[test]
    fn parse_create_function() {
        let stmt = parse_single_statement(
            "CREATE FUNCTION add_one(x INTEGER) RETURNS INTEGER AS 'SELECT x + 1'",
        )
        .unwrap();
        match stmt {
            Statement::CreateFunction(create) => {
                assert_eq!(create.name.to_string(), "add_one");
                assert_eq!(create.parameters.len(), 1);
                assert_eq!(create.parameters[0].name.as_ref().unwrap().name, "x");
                assert!(matches!(create.returns, DataType::Integer));
            }
            _ => panic!("expected CREATE FUNCTION"),
        }
    }

    #[test]
    fn parse_create_function_or_replace() {
        let stmt = parse_single_statement(
            "CREATE OR REPLACE FUNCTION double_val(n INTEGER) RETURNS INTEGER AS 'SELECT n * 2'",
        )
        .unwrap();
        match stmt {
            Statement::CreateFunction(create) => {
                assert!(create.or_replace);
                assert_eq!(create.name.to_string(), "double_val");
            }
            _ => panic!("expected CREATE FUNCTION"),
        }
    }

    #[test]
    fn parse_create_function_with_language() {
        let stmt = parse_single_statement(
            "CREATE FUNCTION myfunc() RETURNS INTEGER LANGUAGE SQL AS 'SELECT 1'",
        )
        .unwrap();
        match stmt {
            Statement::CreateFunction(create) => {
                assert_eq!(create.language, FunctionLanguage::Sql);
            }
            _ => panic!("expected CREATE FUNCTION"),
        }
    }

    #[test]
    fn parse_create_function_immutable() {
        let stmt = parse_single_statement(
            "CREATE FUNCTION square(x INTEGER) RETURNS INTEGER IMMUTABLE AS 'SELECT x * x'",
        )
        .unwrap();
        match stmt {
            Statement::CreateFunction(create) => {
                assert_eq!(create.volatility, Some(FunctionVolatility::Immutable));
            }
            _ => panic!("expected CREATE FUNCTION"),
        }
    }

    #[test]
    fn parse_drop_function() {
        let stmt = parse_single_statement("DROP FUNCTION myfunc").unwrap();
        match stmt {
            Statement::DropFunction(drop) => {
                assert_eq!(drop.name.to_string(), "myfunc");
                assert!(!drop.if_exists);
                assert!(!drop.cascade);
            }
            _ => panic!("expected DROP FUNCTION"),
        }
    }

    #[test]
    fn parse_drop_function_if_exists() {
        let stmt = parse_single_statement("DROP FUNCTION IF EXISTS myfunc").unwrap();
        match stmt {
            Statement::DropFunction(drop) => {
                assert_eq!(drop.name.to_string(), "myfunc");
                assert!(drop.if_exists);
            }
            _ => panic!("expected DROP FUNCTION"),
        }
    }

    #[test]
    fn parse_drop_function_with_args() {
        let stmt = parse_single_statement("DROP FUNCTION myfunc(INTEGER, VARCHAR)").unwrap();
        match stmt {
            Statement::DropFunction(drop) => {
                assert_eq!(drop.name.to_string(), "myfunc");
                assert_eq!(drop.arg_types.len(), 2);
            }
            _ => panic!("expected DROP FUNCTION"),
        }
    }

    // =====================================================
    // Tests for Trigger DDL statements
    // =====================================================

    #[test]
    fn parse_create_trigger_before_insert() {
        let stmt = parse_single_statement(
            "CREATE TRIGGER audit_trigger BEFORE INSERT ON users EXECUTE FUNCTION audit_func()",
        )
        .unwrap();
        match stmt {
            Statement::CreateTrigger(create) => {
                assert_eq!(create.name.name, "audit_trigger");
                assert_eq!(create.timing, TriggerTiming::Before);
                assert_eq!(create.events.len(), 1);
                assert!(matches!(create.events[0], TriggerEvent::Insert));
                assert_eq!(create.table.to_string(), "users");
            }
            _ => panic!("expected CREATE TRIGGER"),
        }
    }

    #[test]
    fn parse_create_trigger_after_update() {
        let stmt = parse_single_statement(
            "CREATE TRIGGER log_changes AFTER UPDATE ON orders EXECUTE FUNCTION log_func()",
        )
        .unwrap();
        match stmt {
            Statement::CreateTrigger(create) => {
                assert_eq!(create.name.name, "log_changes");
                assert_eq!(create.timing, TriggerTiming::After);
                assert!(matches!(create.events[0], TriggerEvent::Update(_)));
            }
            _ => panic!("expected CREATE TRIGGER"),
        }
    }

    #[test]
    fn parse_create_trigger_or_replace() {
        let stmt = parse_single_statement(
            "CREATE OR REPLACE TRIGGER my_trigger AFTER DELETE ON items EXECUTE FUNCTION cleanup()",
        )
        .unwrap();
        match stmt {
            Statement::CreateTrigger(create) => {
                assert!(create.or_replace);
                assert_eq!(create.timing, TriggerTiming::After);
                assert!(matches!(create.events[0], TriggerEvent::Delete));
            }
            _ => panic!("expected CREATE TRIGGER"),
        }
    }

    #[test]
    fn parse_drop_trigger() {
        let stmt = parse_single_statement("DROP TRIGGER my_trigger ON users").unwrap();
        match stmt {
            Statement::DropTrigger(drop) => {
                assert_eq!(drop.name.name, "my_trigger");
                assert_eq!(drop.table.to_string(), "users");
                assert!(!drop.if_exists);
                assert!(!drop.cascade);
            }
            _ => panic!("expected DROP TRIGGER"),
        }
    }

    #[test]
    fn parse_drop_trigger_if_exists() {
        let stmt = parse_single_statement("DROP TRIGGER IF EXISTS my_trigger ON users").unwrap();
        match stmt {
            Statement::DropTrigger(drop) => {
                assert!(drop.if_exists);
                assert_eq!(drop.name.name, "my_trigger");
            }
            _ => panic!("expected DROP TRIGGER"),
        }
    }

    // ========================================================================
    // LATERAL Subquery Parsing Tests
    // ========================================================================

    #[test]
    fn parse_lateral_subquery_basic() {
        let stmt = parse_single_statement(
            "SELECT d.name, e.name FROM departments d, LATERAL (SELECT name FROM employees WHERE department_id = d.id) AS e"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.from.len(), 2);
                // First is a table
                assert!(matches!(select.from[0], TableRef::Table { .. }));
                // Second should be a LateralSubquery
                assert!(matches!(select.from[1], TableRef::LateralSubquery { .. }));
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_lateral_subquery_with_limit() {
        let stmt = parse_single_statement(
            "SELECT * FROM users u, LATERAL (SELECT * FROM orders o WHERE o.user_id = u.id ORDER BY date DESC LIMIT 5) AS recent_orders"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.from.len(), 2);
                if let TableRef::LateralSubquery { query, alias } = &select.from[1] {
                    assert_eq!(alias.name.name, "recent_orders");
                    // Check limit is in subquery
                    assert!(query.limit.is_some());
                } else {
                    panic!("expected LateralSubquery");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_lateral_first_in_from() {
        let stmt = parse_single_statement("SELECT x.n FROM LATERAL (SELECT 1 AS n) AS x").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.from.len(), 1);
                assert!(matches!(select.from[0], TableRef::LateralSubquery { .. }));
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multiple_lateral_subqueries() {
        let stmt = parse_single_statement(
            "SELECT * FROM t1, LATERAL (SELECT * FROM t2 WHERE t2.x = t1.x) AS a, LATERAL (SELECT * FROM t3 WHERE t3.y = a.y) AS b"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.from.len(), 3);
                assert!(matches!(select.from[0], TableRef::Table { .. }));
                assert!(matches!(select.from[1], TableRef::LateralSubquery { .. }));
                assert!(matches!(select.from[2], TableRef::LateralSubquery { .. }));
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_lateral_vs_regular_subquery() {
        // Regular subquery (not lateral)
        let stmt1 =
            parse_single_statement("SELECT * FROM users u, (SELECT 1 AS n) AS nums").unwrap();
        // LATERAL subquery
        let stmt2 =
            parse_single_statement("SELECT * FROM users u, LATERAL (SELECT 1 AS n) AS nums")
                .unwrap();

        match (stmt1, stmt2) {
            (Statement::Select(s1), Statement::Select(s2)) => {
                // Regular subquery should be Subquery variant
                assert!(matches!(s1.from[1], TableRef::Subquery { .. }));
                // LATERAL should be LateralSubquery variant
                assert!(matches!(s2.from[1], TableRef::LateralSubquery { .. }));
            }
            _ => panic!("expected SELECT statements"),
        }
    }
}
