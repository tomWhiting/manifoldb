//! SQL parser implementation.
//!
//! This module provides the core SQL parsing functionality using `sqlparser-rs`
//! as the foundation, with custom transformations to our AST types.

use sqlparser::ast as sp;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::ast::{
    Assignment, BinaryOp, CaseExpr, ColumnConstraint, ColumnDef, ConflictAction, ConflictTarget,
    CreateIndexStatement, CreateTableStatement, DataType, DeleteStatement, DropIndexStatement,
    DropTableStatement, Expr, FunctionCall, Identifier, IndexColumn, InsertSource, InsertStatement,
    JoinClause, JoinCondition, JoinType, Literal, OnConflict, OrderByExpr, ParameterRef,
    QualifiedName, SelectItem, SelectStatement, SetOperation, SetOperator, Statement, TableAlias,
    TableConstraint, TableRef, UnaryOp, UpdateStatement, WindowFrame, WindowFrameBound,
    WindowFrameUnits, WindowSpec,
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

    statements
        .into_iter()
        .map(convert_statement)
        .collect()
}

/// Parses a single SQL statement.
///
/// # Errors
///
/// Returns an error if the SQL is invalid or contains multiple statements.
pub fn parse_single_statement(sql: &str) -> ParseResult<Statement> {
    let mut stmts = parse_sql(sql)?;
    if stmts.len() != 1 {
        return Err(ParseError::SqlSyntax(format!(
            "expected 1 statement, found {}",
            stmts.len()
        )));
    }
    // SAFETY: We just verified there's exactly one statement
    Ok(stmts.remove(0))
}

/// Converts a sqlparser Statement to our Statement.
fn convert_statement(stmt: sp::Statement) -> ParseResult<Statement> {
    match stmt {
        sp::Statement::Query(query) => {
            let select = convert_query(*query)?;
            Ok(Statement::Select(select))
        }
        sp::Statement::Insert(insert) => {
            let insert_stmt = convert_insert(insert)?;
            Ok(Statement::Insert(insert_stmt))
        }
        sp::Statement::Update { table, assignments, from, selection, returning } => {
            let from_vec = from.map(|t| vec![t]);
            let update_stmt = convert_update(table, assignments, from_vec, selection, returning)?;
            Ok(Statement::Update(update_stmt))
        }
        sp::Statement::Delete(delete) => {
            let delete_stmt = convert_delete(delete)?;
            Ok(Statement::Delete(delete_stmt))
        }
        sp::Statement::CreateTable(create) => {
            let create_stmt = convert_create_table(create)?;
            Ok(Statement::CreateTable(create_stmt))
        }
        sp::Statement::CreateIndex(create) => {
            let create_stmt = convert_create_index(create)?;
            Ok(Statement::CreateIndex(create_stmt))
        }
        sp::Statement::Drop { object_type, if_exists, names, cascade, .. } => {
            match object_type {
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
                _ => Err(ParseError::Unsupported(format!("DROP {object_type:?}"))),
            }
        }
        sp::Statement::Explain { statement, .. } => {
            let inner = convert_statement(*statement)?;
            Ok(Statement::Explain(Box::new(inner)))
        }
        _ => Err(ParseError::Unsupported(format!("statement type: {stmt:?}"))),
    }
}

/// Converts a sqlparser Query to our `SelectStatement`.
fn convert_query(query: sp::Query) -> ParseResult<SelectStatement> {
    // Handle WITH clause if present
    if query.with.is_some() {
        return Err(ParseError::Unsupported("WITH clause".to_string()));
    }

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
                    sp::SetOperator::Except => SetOperator::Except,
                },
                all: matches!(set_quantifier, sp::SetQuantifier::All),
                right: right_stmt,
            };
            base.set_op = Some(Box::new(set_op));
            base
        }
        sp::SetExpr::Values(values) => {
            // VALUES as a standalone select
            let rows: Vec<Vec<Expr>> = values.rows
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
        result.order_by = order_by.exprs
            .into_iter()
            .map(convert_order_by_expr)
            .collect::<ParseResult<Vec<_>>>()?;
    }

    if let Some(limit_expr) = query.limit {
        result.limit = Some(convert_expr(limit_expr)?);
    }

    if let Some(offset_expr) = query.offset {
        result.offset = Some(convert_expr(offset_expr.value)?);
    }

    Ok(result)
}

/// Converts a sqlparser Select to our `SelectStatement`.
fn convert_select(select: sp::Select) -> ParseResult<SelectStatement> {
    let distinct = match select.distinct {
        Some(sp::Distinct::Distinct) => true,
        Some(sp::Distinct::On(_)) => return Err(ParseError::Unsupported("DISTINCT ON".to_string())),
        None => false,
    };

    let projection = select.projection
        .into_iter()
        .map(convert_select_item)
        .collect::<ParseResult<Vec<_>>>()?;

    let from = select.from
        .into_iter()
        .map(convert_table_with_joins)
        .collect::<ParseResult<Vec<_>>>()?;

    let where_clause = select.selection
        .map(convert_expr)
        .transpose()?;

    let group_by = match select.group_by {
        sp::GroupByExpr::Expressions(exprs, _) => exprs
            .into_iter()
            .map(convert_expr)
            .collect::<ParseResult<Vec<_>>>()?,
        sp::GroupByExpr::All(_) => return Err(ParseError::Unsupported("GROUP BY ALL".to_string())),
    };

    let having = select.having
        .map(convert_expr)
        .transpose()?;

    Ok(SelectStatement {
        distinct,
        projection,
        from,
        match_clause: None, // Handled separately by extension parser
        where_clause,
        group_by,
        having,
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
            Ok(SelectItem::Expr {
                expr: convert_expr(expr)?,
                alias: None,
            })
        }
        sp::SelectItem::ExprWithAlias { expr, alias } => {
            Ok(SelectItem::Expr {
                expr: convert_expr(expr)?,
                alias: Some(convert_ident(alias)),
            })
        }
        sp::SelectItem::Wildcard(_) => Ok(SelectItem::Wildcard),
        sp::SelectItem::QualifiedWildcard(name, _) => {
            Ok(SelectItem::QualifiedWildcard(convert_object_name(name)))
        }
    }
}

/// Converts a table with joins.
fn convert_table_with_joins(twj: sp::TableWithJoins) -> ParseResult<TableRef> {
    let mut result = convert_table_factor(twj.relation)?;

    for join in twj.joins {
        let right = convert_table_factor(join.relation)?;
        let join_type = match join.join_operator {
            sp::JoinOperator::Inner(_) => JoinType::Inner,
            sp::JoinOperator::LeftOuter(_) => JoinType::LeftOuter,
            sp::JoinOperator::RightOuter(_) => JoinType::RightOuter,
            sp::JoinOperator::FullOuter(_) => JoinType::FullOuter,
            sp::JoinOperator::CrossJoin => JoinType::Cross,
            sp::JoinOperator::LeftSemi(_) | sp::JoinOperator::RightSemi(_) => {
                return Err(ParseError::Unsupported("SEMI JOIN".to_string()));
            }
            sp::JoinOperator::LeftAnti(_) | sp::JoinOperator::RightAnti(_) => {
                return Err(ParseError::Unsupported("ANTI JOIN".to_string()));
            }
            sp::JoinOperator::AsOf { .. } => {
                return Err(ParseError::Unsupported("AS OF JOIN".to_string()));
            }
            sp::JoinOperator::CrossApply | sp::JoinOperator::OuterApply => {
                return Err(ParseError::Unsupported("APPLY".to_string()));
            }
        };

        let condition = match join.join_operator {
            sp::JoinOperator::Inner(constraint)
            | sp::JoinOperator::LeftOuter(constraint)
            | sp::JoinOperator::RightOuter(constraint)
            | sp::JoinOperator::FullOuter(constraint) => convert_join_constraint(constraint)?,
            // All other join types have no condition
            _ => JoinCondition::None,
        };

        result = TableRef::Join(Box::new(JoinClause {
            left: result,
            right,
            join_type,
            condition,
        }));
    }

    Ok(result)
}

/// Converts a join constraint.
fn convert_join_constraint(constraint: sp::JoinConstraint) -> ParseResult<JoinCondition> {
    match constraint {
        sp::JoinConstraint::On(expr) => Ok(JoinCondition::On(convert_expr(expr)?)),
        sp::JoinConstraint::Using(idents) => {
            Ok(JoinCondition::Using(idents.into_iter().map(convert_ident).collect()))
        }
        sp::JoinConstraint::Natural => Ok(JoinCondition::Natural),
        sp::JoinConstraint::None => Ok(JoinCondition::None),
    }
}

/// Converts a table factor.
fn convert_table_factor(factor: sp::TableFactor) -> ParseResult<TableRef> {
    match factor {
        sp::TableFactor::Table { name, alias, .. } => {
            Ok(TableRef::Table {
                name: convert_object_name(name),
                alias: alias.map(convert_table_alias),
            })
        }
        sp::TableFactor::Derived { subquery, alias, .. } => {
            let alias = alias.ok_or_else(|| ParseError::MissingClause("alias for subquery".to_string()))?;
            Ok(TableRef::Subquery {
                query: Box::new(convert_query(*subquery)?),
                alias: convert_table_alias(alias),
            })
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
                    TableRef::Table { alias: ref mut a, .. } => *a = Some(convert_table_alias(alias)),
                    TableRef::Subquery { alias: ref mut a, .. } => *a = convert_table_alias(alias),
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
        sp::FunctionArguments::List(arg_list) => {
            arg_list.args
                .into_iter()
                .map(|arg| match arg {
                    sp::FunctionArg::Unnamed(expr) => expr,
                    sp::FunctionArg::Named { arg, .. } => arg,
                })
                .map(|arg_expr| match arg_expr {
                    sp::FunctionArgExpr::Expr(e) => convert_expr(e),
                    sp::FunctionArgExpr::QualifiedWildcard(name) => {
                        Ok(Expr::QualifiedWildcard(convert_object_name(name)))
                    }
                    sp::FunctionArgExpr::Wildcard => Ok(Expr::Wildcard),
                })
                .collect::<ParseResult<Vec<_>>>()
        }
    }
}

/// Converts a table alias.
fn convert_table_alias(alias: sp::TableAlias) -> TableAlias {
    TableAlias {
        name: convert_ident(alias.name),
        columns: alias.columns.into_iter().map(convert_ident).collect(),
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
            Ok(Expr::Column(QualifiedName::new(
                idents.into_iter().map(convert_ident).collect(),
            )))
        }
        sp::Expr::Value(value) => convert_value(value),
        sp::Expr::BinaryOp { left, op, right } => {
            let left = convert_expr(*left)?;
            let right = convert_expr(*right)?;
            let op = convert_binary_op(&op)?;
            Ok(Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
        sp::Expr::UnaryOp { op, expr } => {
            let operand = convert_expr(*expr)?;
            let op = convert_unary_op(op)?;
            Ok(Expr::UnaryOp {
                op,
                operand: Box::new(operand),
            })
        }
        sp::Expr::Nested(inner) => convert_expr(*inner),
        sp::Expr::Function(func) => convert_function(func),
        sp::Expr::Cast { expr, data_type, .. } => {
            Ok(Expr::Cast {
                expr: Box::new(convert_expr(*expr)?),
                data_type: format_data_type(&data_type),
            })
        }
        sp::Expr::Case { operand, conditions, results, else_result } => {
            let when_clauses: Vec<(Expr, Expr)> = conditions
                .into_iter()
                .zip(results)
                .map(|(cond, result)| Ok((convert_expr(cond)?, convert_expr(result)?)))
                .collect::<ParseResult<Vec<_>>>()?;

            Ok(Expr::Case(CaseExpr {
                operand: operand.map(|e| convert_expr(*e)).transpose()?.map(Box::new),
                when_clauses,
                else_result: else_result.map(|e| convert_expr(*e)).transpose()?.map(Box::new),
            }))
        }
        sp::Expr::Subquery(query) => {
            Ok(Expr::Subquery(crate::ast::expr::Subquery {
                query: Box::new(convert_query(*query)?),
            }))
        }
        sp::Expr::Exists { subquery, .. } => {
            Ok(Expr::Exists(crate::ast::expr::Subquery {
                query: Box::new(convert_query(*subquery)?),
            }))
        }
        sp::Expr::InList { expr, list, negated } => {
            Ok(Expr::InList {
                expr: Box::new(convert_expr(*expr)?),
                list: list.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?,
                negated,
            })
        }
        sp::Expr::InSubquery { expr, subquery, negated } => {
            Ok(Expr::InSubquery {
                expr: Box::new(convert_expr(*expr)?),
                subquery: crate::ast::expr::Subquery {
                    query: Box::new(convert_query(*subquery)?),
                },
                negated,
            })
        }
        sp::Expr::Between { expr, low, high, negated } => {
            Ok(Expr::Between {
                expr: Box::new(convert_expr(*expr)?),
                low: Box::new(convert_expr(*low)?),
                high: Box::new(convert_expr(*high)?),
                negated,
            })
        }
        sp::Expr::IsNull(expr) => {
            Ok(Expr::UnaryOp {
                op: UnaryOp::IsNull,
                operand: Box::new(convert_expr(*expr)?),
            })
        }
        sp::Expr::IsNotNull(expr) => {
            Ok(Expr::UnaryOp {
                op: UnaryOp::IsNotNull,
                operand: Box::new(convert_expr(*expr)?),
            })
        }
        sp::Expr::Tuple(exprs) => {
            Ok(Expr::Tuple(
                exprs.into_iter().map(convert_expr).collect::<ParseResult<Vec<_>>>()?
            ))
        }
        sp::Expr::Array(arr) => {
            let sp::Array { elem, .. } = arr;
            let elements = elem
                .into_iter()
                .map(convert_expr)
                .collect::<ParseResult<Vec<_>>>()?;
            Ok(Expr::Tuple(elements))
        }
        sp::Expr::Subscript { expr, subscript } => {
            match *subscript {
                sp::Subscript::Index { index } => {
                    Ok(Expr::ArrayIndex {
                        array: Box::new(convert_expr(*expr)?),
                        index: Box::new(convert_expr(index)?),
                    })
                }
                sp::Subscript::Slice { .. } => Err(ParseError::Unsupported("subscript slice".to_string())),
            }
        }
        sp::Expr::Like { negated, expr, pattern, escape_char: _, any: _ } => {
            Ok(Expr::BinaryOp {
                left: Box::new(convert_expr(*expr)?),
                op: if negated { BinaryOp::NotLike } else { BinaryOp::Like },
                right: Box::new(convert_expr(*pattern)?),
            })
        }
        sp::Expr::ILike { negated, expr, pattern, escape_char: _, any: _ } => {
            Ok(Expr::BinaryOp {
                left: Box::new(convert_expr(*expr)?),
                op: if negated { BinaryOp::NotILike } else { BinaryOp::ILike },
                right: Box::new(convert_expr(*pattern)?),
            })
        }
        sp::Expr::Named { name, .. } => {
            // Named parameter like $name
            Ok(Expr::Parameter(ParameterRef::Named(name.value)))
        }
        // Handle placeholder for positional parameters
        _ => Err(ParseError::Unsupported(format!("expression type: {expr:?}"))),
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
        sp::BinaryOperator::HashLongArrow => Err(ParseError::Unsupported("#>> operator".to_string())),
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

    let filter = func.filter
        .map(|f| convert_expr(*f))
        .transpose()?
        .map(Box::new);

    let over = func.over
        .map(convert_window_spec)
        .transpose()?;

    Ok(Expr::Function(FunctionCall {
        name,
        args,
        distinct: false, // sqlparser 0.52 handles this differently
        filter,
        over,
    }))
}

/// Converts a window specification.
fn convert_window_spec(spec: sp::WindowType) -> ParseResult<WindowSpec> {
    match spec {
        sp::WindowType::WindowSpec(spec) => {
            let partition_by = spec.partition_by
                .into_iter()
                .map(convert_expr)
                .collect::<ParseResult<Vec<_>>>()?;

            let order_by = spec.order_by
                .into_iter()
                .map(convert_order_by_expr)
                .collect::<ParseResult<Vec<_>>>()?;

            let frame = spec.window_frame
                .map(convert_window_frame)
                .transpose()?;

            Ok(WindowSpec {
                partition_by,
                order_by,
                frame,
            })
        }
        sp::WindowType::NamedWindow(_) => {
            Err(ParseError::Unsupported("named window reference".to_string()))
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
    let end = frame.end_bound
        .map(convert_window_frame_bound)
        .transpose()?;

    Ok(WindowFrame { units, start, end })
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
    let asc = expr.asc.unwrap_or(true); // Default to ASC

    Ok(OrderByExpr {
        expr: Box::new(convert_expr(expr.expr)?),
        asc,
        nulls_first: expr.nulls_first,
    })
}

/// Converts an INSERT statement.
fn convert_insert(insert: sp::Insert) -> ParseResult<InsertStatement> {
    // Extract table name
    let table = convert_object_name(insert.table_name);

    let columns: Vec<Identifier> = insert.columns
        .into_iter()
        .map(convert_ident)
        .collect();

    let source = match insert.source {
        Some(source) => match *source.body {
            sp::SetExpr::Values(values) => {
                let rows: Vec<Vec<Expr>> = values.rows
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
        }
        None => InsertSource::DefaultValues,
    };

    let on_conflict = insert.on
        .map(convert_on_conflict)
        .transpose()?;

    let returning = insert.returning
        .map(|items| items.into_iter().map(convert_select_item).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    Ok(InsertStatement {
        table,
        columns,
        source,
        on_conflict,
        returning,
    })
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
                    ConflictTarget::Constraint(convert_object_name(name).parts.into_iter().next()
                        .unwrap_or_else(|| Identifier::new("unknown")))
                }
                None => ConflictTarget::Columns(vec![]),
            };

            let action = match conflict.action {
                sp::OnConflictAction::DoNothing => ConflictAction::DoNothing,
                sp::OnConflictAction::DoUpdate(update) => ConflictAction::DoUpdate {
                    assignments: update.assignments
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

/// Converts an assignment (for UPDATE or ON CONFLICT).
fn convert_assignment(assign: sp::Assignment) -> ParseResult<Assignment> {
    // Convert assignment target to column name
    let column = match assign.target {
        sp::AssignmentTarget::ColumnName(names) => {
            names.0.into_iter()
                .next()
                .map(convert_ident)
                .ok_or_else(|| ParseError::MissingClause("assignment target".to_string()))?
        }
        sp::AssignmentTarget::Tuple(_) => {
            return Err(ParseError::Unsupported("tuple assignment target".to_string()));
        }
    };

    Ok(Assignment {
        column,
        value: convert_expr(assign.value)?,
    })
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

    let assignments = assignments
        .into_iter()
        .map(convert_assignment)
        .collect::<ParseResult<Vec<_>>>()?;

    let from_clause = from
        .map(|f| f.into_iter()
            .map(convert_table_with_joins)
            .collect::<ParseResult<Vec<_>>>())
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
        sp::FromTable::WithFromKeyword(tables) => {
            tables.into_iter().next()
                .ok_or_else(|| ParseError::MissingClause("FROM".to_string()))?
        }
        sp::FromTable::WithoutKeyword(tables) => {
            tables.into_iter().next()
                .ok_or_else(|| ParseError::MissingClause("table".to_string()))?
        }
    };

    let table_ref = convert_table_with_joins(from_table)?;
    let TableRef::Table { name: table_name, alias } = table_ref else {
        return Err(ParseError::Unsupported("complex DELETE target".to_string()));
    };

    let using = delete.using
        .map(|u| u.into_iter().map(convert_table_with_joins).collect::<ParseResult<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    let where_clause = delete.selection.map(convert_expr).transpose()?;

    let returning = delete.returning
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
    let columns = create.columns
        .into_iter()
        .map(convert_column_def)
        .collect::<ParseResult<Vec<_>>>()?;

    let constraints = create.constraints
        .into_iter()
        .map(convert_table_constraint)
        .collect::<ParseResult<Vec<_>>>()?;

    Ok(CreateTableStatement {
        if_not_exists: create.if_not_exists,
        name: convert_object_name(create.name),
        columns,
        constraints,
    })
}

/// Converts a column definition.
fn convert_column_def(col: sp::ColumnDef) -> ParseResult<ColumnDef> {
    let constraints = col.options
        .into_iter()
        .filter_map(|opt| convert_column_option(opt.option).ok())
        .collect();

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
        sp::ColumnOption::Unique { is_primary, .. } => {
            if is_primary {
                Ok(ColumnConstraint::PrimaryKey)
            } else {
                Ok(ColumnConstraint::Unique)
            }
        }
        sp::ColumnOption::ForeignKey { foreign_table, referred_columns, .. } => {
            Ok(ColumnConstraint::References {
                table: convert_object_name(foreign_table),
                column: referred_columns.into_iter().next().map(convert_ident),
            })
        }
        sp::ColumnOption::Check(expr) => {
            Ok(ColumnConstraint::Check(convert_expr(expr)?))
        }
        sp::ColumnOption::Default(expr) => {
            Ok(ColumnConstraint::Default(convert_expr(expr)?))
        }
        _ => Err(ParseError::Unsupported("column option".to_string())),
    }
}

/// Converts a table constraint.
fn convert_table_constraint(constraint: sp::TableConstraint) -> ParseResult<TableConstraint> {
    match constraint {
        sp::TableConstraint::PrimaryKey { columns, name, .. } => {
            Ok(TableConstraint::PrimaryKey {
                name: name.map(convert_ident),
                columns: columns.into_iter().map(convert_ident).collect(),
            })
        }
        sp::TableConstraint::Unique { columns, name, .. } => {
            Ok(TableConstraint::Unique {
                name: name.map(convert_ident),
                columns: columns.into_iter().map(convert_ident).collect(),
            })
        }
        sp::TableConstraint::ForeignKey { columns, foreign_table, referred_columns, name, .. } => {
            Ok(TableConstraint::ForeignKey {
                name: name.map(convert_ident),
                columns: columns.into_iter().map(convert_ident).collect(),
                references_table: convert_object_name(foreign_table),
                references_columns: referred_columns.into_iter().map(convert_ident).collect(),
            })
        }
        sp::TableConstraint::Check { name, expr } => {
            Ok(TableConstraint::Check {
                name: name.map(convert_ident),
                expr: convert_expr(*expr)?,
            })
        }
        _ => Err(ParseError::Unsupported("table constraint".to_string())),
    }
}

/// Converts a CREATE INDEX statement.
fn convert_create_index(create: sp::CreateIndex) -> ParseResult<CreateIndexStatement> {
    let name = create.name
        .map(convert_object_name)
        .and_then(|n| n.parts.into_iter().next())
        .ok_or_else(|| ParseError::MissingClause("index name".to_string()))?;

    let table = convert_object_name(create.table_name);

    let columns = create.columns
        .into_iter()
        .map(|col| {
            Ok(IndexColumn {
                expr: convert_expr(col.expr)?,
                asc: col.asc,
                nulls_first: col.nulls_first,
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
        using: create.using.map(convert_ident).map(|i| i.name),
        with: vec![],
        where_clause: create.predicate.map(convert_expr).transpose()?,
    })
}

/// Converts a data type.
#[allow(clippy::cast_possible_truncation)]
fn convert_data_type(dt: sp::DataType) -> ParseResult<DataType> {
    match dt {
        sp::DataType::Boolean | sp::DataType::Bool => Ok(DataType::Boolean),
        sp::DataType::SmallInt(_) | sp::DataType::Int2(_) => Ok(DataType::SmallInt),
        sp::DataType::Int(_) | sp::DataType::Integer(_) | sp::DataType::Int4(_) => Ok(DataType::Integer),
        sp::DataType::BigInt(_) | sp::DataType::Int8(_) => Ok(DataType::BigInt),
        sp::DataType::Real | sp::DataType::Float4 => Ok(DataType::Real),
        sp::DataType::DoublePrecision | sp::DataType::Double | sp::DataType::Float8 => Ok(DataType::DoublePrecision),
        sp::DataType::Numeric(info) | sp::DataType::Decimal(info) => {
            let (precision, scale) = match info {
                sp::ExactNumberInfo::None => (None, None),
                sp::ExactNumberInfo::Precision(p) => (Some(p as u32), None),
                sp::ExactNumberInfo::PrecisionAndScale(p, s) => (Some(p as u32), Some(s as u32)),
            };
            Ok(DataType::Numeric { precision, scale })
        }
        sp::DataType::Varchar(len) | sp::DataType::CharVarying(len) => {
            let len_val = len.and_then(|l| {
                match l {
                    sp::CharacterLength::IntegerLength { length, .. } => Some(length as u32),
                    sp::CharacterLength::Max => None,
                }
            });
            Ok(DataType::Varchar(len_val))
        }
        sp::DataType::Text => Ok(DataType::Text),
        sp::DataType::Bytea => Ok(DataType::Bytea),
        sp::DataType::Timestamp(_, _) => Ok(DataType::Timestamp),
        sp::DataType::Date => Ok(DataType::Date),
        sp::DataType::Time(_, _) => Ok(DataType::Time),
        sp::DataType::Interval => Ok(DataType::Interval),
        sp::DataType::JSON => Ok(DataType::Json),
        sp::DataType::Uuid => Ok(DataType::Uuid),
        sp::DataType::Array(elem) => {
            match elem {
                sp::ArrayElemTypeDef::AngleBracket(inner) |
                sp::ArrayElemTypeDef::SquareBracket(inner, _) => {
                    Ok(DataType::Array(Box::new(convert_data_type(*inner)?)))
                }
                sp::ArrayElemTypeDef::None => {
                    Err(ParseError::Unsupported("untyped array".to_string()))
                }
                sp::ArrayElemTypeDef::Parenthesis(_) => {
                    Err(ParseError::Unsupported("parenthesized array type".to_string()))
                }
            }
        }
        sp::DataType::Custom(name, _) => {
            let name_str = name.0.iter()
                .map(|p| p.value.clone())
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
    QualifiedName::new(name.0.into_iter().map(convert_ident).collect())
}

/// Converts an identifier.
fn convert_ident(ident: sp::Ident) -> Identifier {
    Identifier {
        name: ident.value,
        quote_style: ident.quote_style,
    }
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
        let stmt = parse_single_statement("INSERT INTO users (name, age) VALUES ('Alice', 30)").unwrap();
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
            "CREATE TABLE users (id BIGINT PRIMARY KEY, name VARCHAR(100) NOT NULL)"
        ).unwrap();
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
            "SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id"
        ).unwrap();
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
}
