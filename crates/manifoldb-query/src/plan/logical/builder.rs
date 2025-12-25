//! Plan builder.
//!
//! This module provides conversion from AST to logical plan.

// Allow long functions - the builder has complex logic that's clearer in one place
#![allow(clippy::too_many_lines)]
// Allow these patterns that are intentional
#![allow(clippy::match_same_arms)]
#![allow(clippy::unnecessary_wraps)]
// Allow unused self - these methods may need self in future for context
#![allow(clippy::unused_self)]
// Allow ref_option - we're passing references to struct fields
#![allow(clippy::ref_option)]
// Allow these casts - we check for negative values explicitly
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// Allow map_unwrap_or - the closure form is clearer here
#![allow(clippy::map_unwrap_or)]

use crate::ast::{
    self, CreateCollectionStatement, CreateIndexStatement, CreateTableStatement, DeleteStatement,
    DropCollectionStatement, DropIndexStatement, DropTableStatement, Expr, GraphPattern,
    InsertSource, InsertStatement, JoinClause, JoinCondition, JoinType as AstJoinType,
    MatchStatement, PathPattern, SelectItem, SelectStatement, SetOperation, SetOperator, Statement,
    TableRef, UpdateStatement,
};

use super::ddl::{
    CreateCollectionNode, CreateIndexNode, CreateTableNode, DropCollectionNode, DropIndexNode,
    DropTableNode,
};

use super::expr::{
    AggregateFunction, HybridCombinationMethod, HybridExprComponent, LogicalExpr, ScalarFunction,
    SortOrder,
};
use super::graph::{ExpandDirection, ExpandLength, ExpandNode};
use super::node::LogicalPlan;
use super::relational::{
    AggregateNode, FilterNode, JoinNode, JoinType, LimitNode, ProjectNode, ScanNode, SetOpNode,
    SetOpType, SortNode, ValuesNode,
};
use super::validate::{PlanError, PlanResult};

/// Builds logical plans from AST.
///
/// # Example
///
/// ```
/// use manifoldb_query::parser::parse_single_statement;
/// use manifoldb_query::plan::logical::PlanBuilder;
///
/// let stmt = parse_single_statement("SELECT * FROM users WHERE id = 1").unwrap();
/// let plan = PlanBuilder::new().build_statement(&stmt).unwrap();
/// ```
#[derive(Debug, Default)]
pub struct PlanBuilder {
    /// Counter for generating unique aliases.
    alias_counter: usize,
}

impl PlanBuilder {
    /// Creates a new plan builder.
    #[must_use]
    pub const fn new() -> Self {
        Self { alias_counter: 0 }
    }

    /// Generates a unique alias.
    fn next_alias(&mut self, prefix: &str) -> String {
        self.alias_counter += 1;
        format!("{}_{}", prefix, self.alias_counter)
    }

    /// Builds a logical plan from a statement.
    pub fn build_statement(&mut self, stmt: &Statement) -> PlanResult<LogicalPlan> {
        match stmt {
            Statement::Select(select) => self.build_select(select),
            Statement::Insert(insert) => self.build_insert(insert),
            Statement::Update(update) => self.build_update(update),
            Statement::Delete(delete) => self.build_delete(delete),
            Statement::Match(match_stmt) => self.build_match(match_stmt),
            Statement::Explain(inner) => {
                // For EXPLAIN, we still build the plan
                self.build_statement(inner)
            }
            Statement::CreateTable(create) => self.build_create_table(create),
            Statement::CreateIndex(create) => self.build_create_index(create),
            Statement::CreateCollection(create) => self.build_create_collection(create),
            Statement::DropTable(drop) => self.build_drop_table(drop),
            Statement::DropIndex(drop) => self.build_drop_index(drop),
            Statement::DropCollection(drop) => self.build_drop_collection(drop),
        }
    }

    /// Builds a logical plan from a standalone MATCH statement.
    ///
    /// This converts the Cypher-style MATCH statement to an equivalent SELECT
    /// and builds the plan from that.
    pub fn build_match(&mut self, match_stmt: &MatchStatement) -> PlanResult<LogicalPlan> {
        // Convert MATCH to SELECT and build using existing infrastructure
        let select = match_stmt.to_select();
        self.build_select(&select)
    }

    /// Builds a logical plan from a SELECT statement.
    pub fn build_select(&mut self, select: &SelectStatement) -> PlanResult<LogicalPlan> {
        // Start with FROM clause
        let mut plan = self.build_from(&select.from)?;

        // Handle MATCH clause for graph patterns
        if let Some(pattern) = &select.match_clause {
            plan = self.build_graph_pattern(plan, pattern)?;
        }

        // Add WHERE clause
        if let Some(where_clause) = &select.where_clause {
            let predicate = self.build_expr(where_clause)?;
            plan = LogicalPlan::Filter { node: FilterNode::new(predicate), input: Box::new(plan) };
        }

        // Handle GROUP BY and aggregates
        let has_aggregates = self.has_aggregates(&select.projection);
        if !select.group_by.is_empty() || has_aggregates {
            plan = self.build_aggregate(plan, select)?;

            // Add HAVING clause
            if let Some(having) = &select.having {
                let predicate = self.build_expr(having)?;
                plan =
                    LogicalPlan::Filter { node: FilterNode::new(predicate), input: Box::new(plan) };
            }
        }

        // Add projection
        let projection = self.build_projection(&select.projection)?;
        plan = LogicalPlan::Project { node: ProjectNode::new(projection), input: Box::new(plan) };

        // Handle DISTINCT
        if select.distinct {
            plan = plan.distinct();
        }

        // Add ORDER BY
        if !select.order_by.is_empty() {
            let order_by = self.build_order_by(&select.order_by)?;
            plan = LogicalPlan::Sort { node: SortNode::new(order_by), input: Box::new(plan) };
        }

        // Add LIMIT/OFFSET
        plan = self.apply_limit_offset(plan, &select.limit, &select.offset)?;

        // Handle set operations (UNION, INTERSECT, EXCEPT)
        if let Some(set_op) = &select.set_op {
            plan = self.build_set_operation(plan, set_op)?;
        }

        Ok(plan)
    }

    /// Builds plan from FROM clause.
    fn build_from(&mut self, from: &[TableRef]) -> PlanResult<LogicalPlan> {
        if from.is_empty() {
            // No FROM clause - return empty relation with single row
            // (for queries like SELECT 1 + 2)
            return Ok(LogicalPlan::Values(ValuesNode::new(vec![vec![]])));
        }

        let mut plan = self.build_table_ref(&from[0])?;

        // Cross join additional tables
        for table in from.iter().skip(1) {
            let right = self.build_table_ref(table)?;
            plan = plan.cross_join(right);
        }

        Ok(plan)
    }

    /// Builds plan from a table reference.
    fn build_table_ref(&mut self, table_ref: &TableRef) -> PlanResult<LogicalPlan> {
        match table_ref {
            TableRef::Table { name, alias } => {
                let table_name =
                    name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

                let mut scan = ScanNode::new(table_name);
                if let Some(a) = alias {
                    scan = scan.with_alias(&a.name.name);
                }

                Ok(LogicalPlan::Scan(Box::new(scan)))
            }

            TableRef::Subquery { query, alias } => {
                let subquery = self.build_select(query)?;
                Ok(subquery.alias(&alias.name.name))
            }

            TableRef::Join(join) => self.build_join(join),

            TableRef::TableFunction { name, args, alias } => {
                // Table functions are handled as scans with special names
                let func_name =
                    name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                let _ = args; // Arguments would be passed to the scan

                let mut scan = ScanNode::new(format!("@function:{func_name}"));
                if let Some(a) = alias {
                    scan = scan.with_alias(&a.name.name);
                }

                Ok(LogicalPlan::Scan(Box::new(scan)))
            }
        }
    }

    /// Builds plan from a JOIN clause.
    fn build_join(&mut self, join: &JoinClause) -> PlanResult<LogicalPlan> {
        let left = self.build_table_ref(&join.left)?;
        let right = self.build_table_ref(&join.right)?;

        let join_type = match join.join_type {
            AstJoinType::Inner => JoinType::Inner,
            AstJoinType::LeftOuter => JoinType::Left,
            AstJoinType::RightOuter => JoinType::Right,
            AstJoinType::FullOuter => JoinType::Full,
            AstJoinType::Cross => JoinType::Cross,
        };

        let node = match &join.condition {
            JoinCondition::On(expr) => {
                let condition = self.build_expr(expr)?;
                JoinNode { join_type, condition: Some(condition), using_columns: vec![] }
            }
            JoinCondition::Using(columns) => {
                let cols = columns.iter().map(|c| c.name.clone()).collect();
                JoinNode { join_type, condition: None, using_columns: cols }
            }
            JoinCondition::Natural => {
                // Natural join - would need schema to resolve
                JoinNode {
                    join_type,
                    condition: None,
                    using_columns: vec![], // Would be filled by analyzer
                }
            }
            JoinCondition::None => JoinNode { join_type, condition: None, using_columns: vec![] },
        };

        Ok(LogicalPlan::Join { node: Box::new(node), left: Box::new(left), right: Box::new(right) })
    }

    /// Builds aggregate plan.
    fn build_aggregate(
        &mut self,
        input: LogicalPlan,
        select: &SelectStatement,
    ) -> PlanResult<LogicalPlan> {
        let group_by: Vec<LogicalExpr> =
            select.group_by.iter().map(|e| self.build_expr(e)).collect::<PlanResult<_>>()?;

        // Extract aggregate expressions from projection - pre-allocate based on projection size
        let mut aggregates = Vec::with_capacity(select.projection.len());
        for item in &select.projection {
            if let SelectItem::Expr { expr, .. } = item {
                self.collect_aggregates(expr, &mut aggregates)?;
            }
        }

        Ok(LogicalPlan::Aggregate {
            node: Box::new(AggregateNode::new(group_by, aggregates)),
            input: Box::new(input),
        })
    }

    /// Collects aggregate expressions from an expression tree.
    fn collect_aggregates(
        &mut self,
        expr: &Expr,
        aggregates: &mut Vec<LogicalExpr>,
    ) -> PlanResult<()> {
        match expr {
            Expr::Function(func) => {
                let name =
                    func.name.parts.last().map(|p| p.name.to_uppercase()).unwrap_or_default();

                if let Some(agg_func) = self.parse_aggregate_function(&name) {
                    let arg = if func.args.is_empty() {
                        LogicalExpr::wildcard()
                    } else {
                        self.build_expr(&func.args[0])?
                    };

                    aggregates.push(LogicalExpr::AggregateFunction {
                        func: agg_func,
                        arg: Box::new(arg),
                        distinct: func.distinct,
                    });
                } else {
                    // Check arguments for nested aggregates
                    for arg in &func.args {
                        self.collect_aggregates(arg, aggregates)?;
                    }
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_aggregates(left, aggregates)?;
                self.collect_aggregates(right, aggregates)?;
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_aggregates(operand, aggregates)?;
            }
            Expr::Case(case) => {
                if let Some(op) = &case.operand {
                    self.collect_aggregates(op, aggregates)?;
                }
                for (when, then) in &case.when_clauses {
                    self.collect_aggregates(when, aggregates)?;
                    self.collect_aggregates(then, aggregates)?;
                }
                if let Some(else_result) = &case.else_result {
                    self.collect_aggregates(else_result, aggregates)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Parses an aggregate function name.
    fn parse_aggregate_function(&self, name: &str) -> Option<AggregateFunction> {
        match name {
            "COUNT" => Some(AggregateFunction::Count),
            "SUM" => Some(AggregateFunction::Sum),
            "AVG" => Some(AggregateFunction::Avg),
            "MIN" => Some(AggregateFunction::Min),
            "MAX" => Some(AggregateFunction::Max),
            "ARRAY_AGG" => Some(AggregateFunction::ArrayAgg),
            "STRING_AGG" => Some(AggregateFunction::StringAgg),
            "VECTOR_AVG" => Some(AggregateFunction::VectorAvg),
            "VECTOR_CENTROID" => Some(AggregateFunction::VectorCentroid),
            _ => None,
        }
    }

    /// Checks if projection contains aggregates.
    fn has_aggregates(&self, projection: &[SelectItem]) -> bool {
        projection.iter().any(|item| {
            if let SelectItem::Expr { expr, .. } = item {
                self.expr_has_aggregate(expr)
            } else {
                false
            }
        })
    }

    /// Checks if an expression contains aggregates.
    fn expr_has_aggregate(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Function(func) => {
                let name =
                    func.name.parts.last().map(|p| p.name.to_uppercase()).unwrap_or_default();
                self.parse_aggregate_function(&name).is_some()
                    || func.args.iter().any(|a| self.expr_has_aggregate(a))
            }
            Expr::BinaryOp { left, right, .. } => {
                self.expr_has_aggregate(left) || self.expr_has_aggregate(right)
            }
            Expr::UnaryOp { operand, .. } => self.expr_has_aggregate(operand),
            _ => false,
        }
    }

    /// Builds projection expressions.
    fn build_projection(&mut self, items: &[SelectItem]) -> PlanResult<Vec<LogicalExpr>> {
        let mut exprs = Vec::with_capacity(items.len());

        for item in items {
            match item {
                SelectItem::Wildcard => {
                    exprs.push(LogicalExpr::Wildcard);
                }
                SelectItem::QualifiedWildcard(name) => {
                    let qualifier =
                        name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                    exprs.push(LogicalExpr::QualifiedWildcard(qualifier));
                }
                SelectItem::Expr { expr, alias } => {
                    let mut e = self.build_expr(expr)?;
                    if let Some(a) = alias {
                        e = e.alias(&a.name);
                    }
                    exprs.push(e);
                }
            }
        }

        Ok(exprs)
    }

    /// Builds ORDER BY expressions.
    fn build_order_by(&mut self, orders: &[ast::OrderByExpr]) -> PlanResult<Vec<SortOrder>> {
        orders
            .iter()
            .map(|o| {
                let expr = self.build_expr(&o.expr)?;
                Ok(SortOrder { expr, ascending: o.asc, nulls_first: o.nulls_first })
            })
            .collect()
    }

    /// Applies LIMIT and OFFSET to a plan.
    fn apply_limit_offset(
        &self,
        plan: LogicalPlan,
        limit: &Option<Expr>,
        offset: &Option<Expr>,
    ) -> PlanResult<LogicalPlan> {
        let limit_val = if let Some(l) = limit { Some(self.expr_to_usize(l)?) } else { None };

        let offset_val = if let Some(o) = offset { Some(self.expr_to_usize(o)?) } else { None };

        if limit_val.is_none() && offset_val.is_none() {
            return Ok(plan);
        }

        Ok(LogicalPlan::Limit {
            node: LimitNode { limit: limit_val, offset: offset_val },
            input: Box::new(plan),
        })
    }

    /// Converts an expression to usize (for LIMIT/OFFSET).
    fn expr_to_usize(&self, expr: &Expr) -> PlanResult<usize> {
        match expr {
            Expr::Literal(ast::Literal::Integer(n)) => {
                if *n >= 0 {
                    Ok(*n as usize)
                } else {
                    Err(PlanError::InvalidLimit("negative value".to_string()))
                }
            }
            _ => Err(PlanError::InvalidLimit("expected integer literal".to_string())),
        }
    }

    /// Builds set operation (UNION, INTERSECT, EXCEPT).
    fn build_set_operation(
        &mut self,
        left: LogicalPlan,
        set_op: &SetOperation,
    ) -> PlanResult<LogicalPlan> {
        let right = self.build_select(&set_op.right)?;

        let op_type = match (set_op.op, set_op.all) {
            (SetOperator::Union, false) => SetOpType::Union,
            (SetOperator::Union, true) => SetOpType::UnionAll,
            (SetOperator::Intersect, false) => SetOpType::Intersect,
            (SetOperator::Intersect, true) => SetOpType::IntersectAll,
            (SetOperator::Except, false) => SetOpType::Except,
            (SetOperator::Except, true) => SetOpType::ExceptAll,
        };

        Ok(LogicalPlan::SetOp {
            node: SetOpNode::new(op_type),
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Builds a graph pattern from a MATCH clause.
    fn build_graph_pattern(
        &mut self,
        input: LogicalPlan,
        pattern: &GraphPattern,
    ) -> PlanResult<LogicalPlan> {
        let mut plan = input;

        for path in &pattern.paths {
            plan = self.build_path_pattern(plan, path)?;
        }

        Ok(plan)
    }

    /// Builds a path pattern.
    fn build_path_pattern(
        &mut self,
        input: LogicalPlan,
        path: &PathPattern,
    ) -> PlanResult<LogicalPlan> {
        let mut plan = input;

        // Start with the starting node
        let start_var = path
            .start
            .variable
            .as_ref()
            .map(|v| v.name.clone())
            .unwrap_or_else(|| self.next_alias("node"));

        // Build expand nodes for each step
        let mut current_var = start_var;

        for (edge, node) in &path.steps {
            let dst_var = node
                .variable
                .as_ref()
                .map(|v| v.name.clone())
                .unwrap_or_else(|| self.next_alias("node"));

            let direction = match edge.direction {
                ast::EdgeDirection::Right => ExpandDirection::Outgoing,
                ast::EdgeDirection::Left => ExpandDirection::Incoming,
                ast::EdgeDirection::Undirected => ExpandDirection::Both,
            };

            let mut expand = ExpandNode::new(&current_var, &dst_var, direction);

            // Add edge types
            if !edge.edge_types.is_empty() {
                expand = expand
                    .with_edge_types(edge.edge_types.iter().map(|t| t.name.clone()).collect());
            }

            // Add edge variable
            if let Some(var) = &edge.variable {
                expand = expand.with_edge_var(&var.name);
            }

            // Add variable length
            expand = expand.with_length(ExpandLength::from_ast(&edge.length));

            // Add node labels
            if !node.labels.is_empty() {
                expand =
                    expand.with_node_labels(node.labels.iter().map(|l| l.name.clone()).collect());
            }

            plan = LogicalPlan::Expand { node: Box::new(expand), input: Box::new(plan) };

            current_var = dst_var;
        }

        Ok(plan)
    }

    /// Builds an INSERT plan.
    fn build_insert(&mut self, insert: &InsertStatement) -> PlanResult<LogicalPlan> {
        let table =
            insert.table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let columns: Vec<String> = insert.columns.iter().map(|c| c.name.clone()).collect();

        let input = match &insert.source {
            InsertSource::Values(rows) => {
                let logical_rows: Vec<Vec<LogicalExpr>> = rows
                    .iter()
                    .map(|row| row.iter().map(|e| self.build_expr(e)).collect::<PlanResult<_>>())
                    .collect::<PlanResult<_>>()?;
                LogicalPlan::Values(ValuesNode::new(logical_rows))
            }
            InsertSource::Query(query) => self.build_select(query)?,
            InsertSource::DefaultValues => LogicalPlan::Values(ValuesNode::new(vec![vec![]])),
        };

        let returning = insert
            .returning
            .iter()
            .map(|item| match item {
                SelectItem::Wildcard => Ok(LogicalExpr::Wildcard),
                SelectItem::QualifiedWildcard(name) => {
                    let qualifier =
                        name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                    Ok(LogicalExpr::QualifiedWildcard(qualifier))
                }
                SelectItem::Expr { expr, alias } => {
                    let mut e = self.build_expr(expr)?;
                    if let Some(a) = alias {
                        e = e.alias(&a.name);
                    }
                    Ok(e)
                }
            })
            .collect::<PlanResult<_>>()?;

        Ok(LogicalPlan::Insert { table, columns, input: Box::new(input), returning })
    }

    /// Builds an UPDATE plan.
    fn build_update(&mut self, update: &UpdateStatement) -> PlanResult<LogicalPlan> {
        let table =
            update.table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let assignments: Vec<(String, LogicalExpr)> = update
            .assignments
            .iter()
            .map(|a| Ok((a.column.name.clone(), self.build_expr(&a.value)?)))
            .collect::<PlanResult<_>>()?;

        let filter =
            if let Some(w) = &update.where_clause { Some(self.build_expr(w)?) } else { None };

        let returning = update
            .returning
            .iter()
            .map(|item| match item {
                SelectItem::Wildcard => Ok(LogicalExpr::Wildcard),
                SelectItem::QualifiedWildcard(name) => {
                    let qualifier =
                        name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                    Ok(LogicalExpr::QualifiedWildcard(qualifier))
                }
                SelectItem::Expr { expr, alias } => {
                    let mut e = self.build_expr(expr)?;
                    if let Some(a) = alias {
                        e = e.alias(&a.name);
                    }
                    Ok(e)
                }
            })
            .collect::<PlanResult<_>>()?;

        Ok(LogicalPlan::Update { table, assignments, filter, returning })
    }

    /// Builds a DELETE plan.
    fn build_delete(&mut self, delete: &DeleteStatement) -> PlanResult<LogicalPlan> {
        let table =
            delete.table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let filter =
            if let Some(w) = &delete.where_clause { Some(self.build_expr(w)?) } else { None };

        let returning = delete
            .returning
            .iter()
            .map(|item| match item {
                SelectItem::Wildcard => Ok(LogicalExpr::Wildcard),
                SelectItem::QualifiedWildcard(name) => {
                    let qualifier =
                        name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                    Ok(LogicalExpr::QualifiedWildcard(qualifier))
                }
                SelectItem::Expr { expr, alias } => {
                    let mut e = self.build_expr(expr)?;
                    if let Some(a) = alias {
                        e = e.alias(&a.name);
                    }
                    Ok(e)
                }
            })
            .collect::<PlanResult<_>>()?;

        Ok(LogicalPlan::Delete { table, filter, returning })
    }

    /// Builds a CREATE TABLE plan.
    fn build_create_table(&self, create: &CreateTableStatement) -> PlanResult<LogicalPlan> {
        let name = create.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = CreateTableNode::new(name, create.columns.clone())
            .with_if_not_exists(create.if_not_exists)
            .with_constraints(create.constraints.clone());

        Ok(LogicalPlan::CreateTable(node))
    }

    /// Builds a DROP TABLE plan.
    fn build_drop_table(&self, drop: &DropTableStatement) -> PlanResult<LogicalPlan> {
        let names: Vec<String> = drop
            .names
            .iter()
            .map(|n| n.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("."))
            .collect();

        let node =
            DropTableNode::new(names).with_if_exists(drop.if_exists).with_cascade(drop.cascade);

        Ok(LogicalPlan::DropTable(node))
    }

    /// Builds a CREATE INDEX plan.
    fn build_create_index(&self, create: &CreateIndexStatement) -> PlanResult<LogicalPlan> {
        let table =
            create.table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = CreateIndexNode::new(create.name.name.clone(), table, create.columns.clone())
            .with_unique(create.unique)
            .with_if_not_exists(create.if_not_exists)
            .with_using(create.using.clone())
            .with_options(create.with.clone())
            .with_where_clause(create.where_clause.clone());

        Ok(LogicalPlan::CreateIndex(node))
    }

    /// Builds a DROP INDEX plan.
    fn build_drop_index(&self, drop: &DropIndexStatement) -> PlanResult<LogicalPlan> {
        let names: Vec<String> = drop
            .names
            .iter()
            .map(|n| n.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("."))
            .collect();

        let node =
            DropIndexNode::new(names).with_if_exists(drop.if_exists).with_cascade(drop.cascade);

        Ok(LogicalPlan::DropIndex(node))
    }

    /// Builds a CREATE COLLECTION plan.
    fn build_create_collection(
        &self,
        create: &CreateCollectionStatement,
    ) -> PlanResult<LogicalPlan> {
        let node = CreateCollectionNode::new(create.name.name.clone(), create.vectors.clone())
            .with_if_not_exists(create.if_not_exists);

        Ok(LogicalPlan::CreateCollection(node))
    }

    /// Builds a DROP COLLECTION plan.
    fn build_drop_collection(&self, drop: &DropCollectionStatement) -> PlanResult<LogicalPlan> {
        let names: Vec<String> = drop.names.iter().map(|n| n.name.clone()).collect();

        let node = DropCollectionNode::new(names)
            .with_if_exists(drop.if_exists)
            .with_cascade(drop.cascade);

        Ok(LogicalPlan::DropCollection(node))
    }

    /// Builds a logical expression from an AST expression.
    pub fn build_expr(&mut self, expr: &Expr) -> PlanResult<LogicalExpr> {
        match expr {
            Expr::Literal(lit) => Ok(LogicalExpr::Literal(lit.clone())),

            Expr::Column(name) => Ok(LogicalExpr::from_qualified_name(name)),

            Expr::Parameter(param) => {
                let idx = match param {
                    ast::ParameterRef::Positional(n) => *n,
                    ast::ParameterRef::Named(_) => 0, // Would need name resolution
                    ast::ParameterRef::Anonymous => 0,
                };
                Ok(LogicalExpr::Parameter(idx))
            }

            Expr::BinaryOp { left, op, right } => {
                let l = self.build_expr(left)?;
                let r = self.build_expr(right)?;
                Ok(LogicalExpr::BinaryOp { left: Box::new(l), op: *op, right: Box::new(r) })
            }

            Expr::UnaryOp { op, operand } => {
                let operand = self.build_expr(operand)?;
                Ok(LogicalExpr::UnaryOp { op: *op, operand: Box::new(operand) })
            }

            Expr::Function(func) => {
                let name =
                    func.name.parts.last().map(|p| p.name.to_uppercase()).unwrap_or_default();

                // Check if it's an aggregate function
                if let Some(agg_func) = self.parse_aggregate_function(&name) {
                    let arg = if func.args.is_empty() {
                        LogicalExpr::Wildcard
                    } else {
                        self.build_expr(&func.args[0])?
                    };

                    return Ok(LogicalExpr::AggregateFunction {
                        func: agg_func,
                        arg: Box::new(arg),
                        distinct: func.distinct,
                    });
                }

                // Check if it's a known scalar function
                let scalar_func = match name.as_str() {
                    "UPPER" => Some(ScalarFunction::Upper),
                    "LOWER" => Some(ScalarFunction::Lower),
                    "LENGTH" => Some(ScalarFunction::Length),
                    "CONCAT" => Some(ScalarFunction::Concat),
                    "SUBSTRING" => Some(ScalarFunction::Substring),
                    "TRIM" => Some(ScalarFunction::Trim),
                    "COALESCE" => Some(ScalarFunction::Coalesce),
                    "NULLIF" => Some(ScalarFunction::NullIf),
                    "ABS" => Some(ScalarFunction::Abs),
                    "CEIL" | "CEILING" => Some(ScalarFunction::Ceil),
                    "FLOOR" => Some(ScalarFunction::Floor),
                    "ROUND" => Some(ScalarFunction::Round),
                    "SQRT" => Some(ScalarFunction::Sqrt),
                    "POWER" | "POW" => Some(ScalarFunction::Power),
                    "NOW" => Some(ScalarFunction::Now),
                    "CURRENT_DATE" => Some(ScalarFunction::CurrentDate),
                    "CURRENT_TIME" => Some(ScalarFunction::CurrentTime),
                    "EXTRACT" => Some(ScalarFunction::Extract),
                    "VECTOR_DIMENSION" => Some(ScalarFunction::VectorDimension),
                    "VECTOR_NORM" => Some(ScalarFunction::VectorNorm),
                    _ => None,
                };

                if let Some(sf) = scalar_func {
                    let args =
                        func.args.iter().map(|a| self.build_expr(a)).collect::<PlanResult<_>>()?;
                    return Ok(LogicalExpr::ScalarFunction { func: sf, args });
                }

                // Unknown function - treat as custom
                let args =
                    func.args.iter().map(|a| self.build_expr(a)).collect::<PlanResult<_>>()?;
                Ok(LogicalExpr::ScalarFunction {
                    func: ScalarFunction::Custom(0), // Would need function registry
                    args,
                })
            }

            Expr::Cast { expr, data_type } => {
                let e = self.build_expr(expr)?;
                // Parse data type string to DataType enum
                let dt = self.parse_data_type(data_type)?;
                Ok(LogicalExpr::Cast { expr: Box::new(e), data_type: dt })
            }

            Expr::Case(case) => {
                let operand = if let Some(op) = &case.operand {
                    Some(Box::new(self.build_expr(op)?))
                } else {
                    None
                };

                let when_clauses = case
                    .when_clauses
                    .iter()
                    .map(|(when, then)| Ok((self.build_expr(when)?, self.build_expr(then)?)))
                    .collect::<PlanResult<_>>()?;

                let else_result = if let Some(e) = &case.else_result {
                    Some(Box::new(self.build_expr(e)?))
                } else {
                    None
                };

                Ok(LogicalExpr::Case { operand, when_clauses, else_result })
            }

            Expr::InList { expr, list, negated } => {
                let e = self.build_expr(expr)?;
                let l = list.iter().map(|i| self.build_expr(i)).collect::<PlanResult<_>>()?;
                Ok(LogicalExpr::InList { expr: Box::new(e), list: l, negated: *negated })
            }

            Expr::Between { expr, low, high, negated } => {
                let e = self.build_expr(expr)?;
                let lo = self.build_expr(low)?;
                let hi = self.build_expr(high)?;
                Ok(LogicalExpr::Between {
                    expr: Box::new(e),
                    low: Box::new(lo),
                    high: Box::new(hi),
                    negated: *negated,
                })
            }

            Expr::Subquery(subquery) => {
                let plan = self.build_select(&subquery.query)?;
                Ok(LogicalExpr::Subquery(Box::new(plan)))
            }

            Expr::Exists(subquery) => {
                let plan = self.build_select(&subquery.query)?;
                Ok(LogicalExpr::Exists { subquery: Box::new(plan), negated: false })
            }

            Expr::InSubquery { expr, subquery, negated } => {
                let e = self.build_expr(expr)?;
                let plan = self.build_select(&subquery.query)?;
                Ok(LogicalExpr::InSubquery {
                    expr: Box::new(e),
                    subquery: Box::new(plan),
                    negated: *negated,
                })
            }

            Expr::Wildcard => Ok(LogicalExpr::Wildcard),

            Expr::QualifiedWildcard(name) => {
                let qualifier =
                    name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                Ok(LogicalExpr::QualifiedWildcard(qualifier))
            }

            Expr::HybridSearch { components, method } => {
                // Convert AST HybridSearch to a logical HybridSearch expression
                // This properly handles all components for weighted sum or RRF combination
                if components.is_empty() {
                    return Ok(LogicalExpr::Literal(ast::Literal::Null));
                }

                // Convert each AST component to a logical component
                let logical_components: Vec<HybridExprComponent> = components
                    .iter()
                    .map(|c| {
                        let distance_expr = self.build_expr(&c.distance_expr)?;
                        Ok(HybridExprComponent::new(distance_expr, c.weight))
                    })
                    .collect::<PlanResult<Vec<_>>>()?;

                // Convert the combination method
                let logical_method = match method {
                    ast::HybridCombinationMethod::WeightedSum => HybridCombinationMethod::WeightedSum,
                    ast::HybridCombinationMethod::RRF { k } => HybridCombinationMethod::RRF { k: *k },
                };

                Ok(LogicalExpr::HybridSearch {
                    components: logical_components,
                    method: logical_method,
                })
            }

            Expr::Tuple(exprs) => {
                // Try to convert tuple of numeric literals to a vector
                // This handles array literals like [0.1, 0.2, 0.3] used for vectors
                let mut vec_elements = Vec::new();
                let mut all_numeric = true;

                for expr in exprs.iter() {
                    match expr {
                        Expr::Literal(ast::Literal::Integer(n)) => {
                            vec_elements.push(*n as f32);
                        }
                        Expr::Literal(ast::Literal::Float(f)) => {
                            vec_elements.push(*f as f32);
                        }
                        _ => {
                            all_numeric = false;
                            break;
                        }
                    }
                }

                if all_numeric && !vec_elements.is_empty() {
                    // Convert numeric array to vector literal
                    Ok(LogicalExpr::Literal(ast::Literal::Vector(vec_elements)))
                } else if let Some(first) = exprs.first() {
                    // Fall back to first element for non-numeric tuples
                    self.build_expr(first)
                } else {
                    Ok(LogicalExpr::Literal(ast::Literal::Null))
                }
            }

            Expr::ArrayIndex { array, index } => {
                // Array indexing could be implemented as a function
                let arr = self.build_expr(array)?;
                let idx = self.build_expr(index)?;
                Ok(LogicalExpr::ScalarFunction {
                    func: ScalarFunction::Custom(0),
                    args: vec![arr, idx],
                })
            }
        }
    }

    /// Parses a data type string.
    fn parse_data_type(&self, type_str: &str) -> PlanResult<ast::DataType> {
        let upper = type_str.to_uppercase();
        let dt = match upper.as_str() {
            "BOOLEAN" | "BOOL" => ast::DataType::Boolean,
            "SMALLINT" | "INT2" => ast::DataType::SmallInt,
            "INTEGER" | "INT" | "INT4" => ast::DataType::Integer,
            "BIGINT" | "INT8" => ast::DataType::BigInt,
            "REAL" | "FLOAT4" => ast::DataType::Real,
            "DOUBLE PRECISION" | "FLOAT8" | "DOUBLE" => ast::DataType::DoublePrecision,
            "TEXT" => ast::DataType::Text,
            "BYTEA" => ast::DataType::Bytea,
            "TIMESTAMP" => ast::DataType::Timestamp,
            "DATE" => ast::DataType::Date,
            "TIME" => ast::DataType::Time,
            "INTERVAL" => ast::DataType::Interval,
            "JSON" => ast::DataType::Json,
            "JSONB" => ast::DataType::Jsonb,
            "UUID" => ast::DataType::Uuid,
            _ if upper.starts_with("VARCHAR") => {
                // Parse VARCHAR(n)
                ast::DataType::Varchar(None)
            }
            _ if upper.starts_with("VECTOR") => {
                // Parse VECTOR(n)
                ast::DataType::Vector(None)
            }
            _ => ast::DataType::Custom(type_str.to_string()),
        };
        Ok(dt)
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parse_single_statement;

    use super::*;

    fn build_query(sql: &str) -> PlanResult<LogicalPlan> {
        let stmt = parse_single_statement(sql).expect("parse failed");
        PlanBuilder::new().build_statement(&stmt)
    }

    #[test]
    fn simple_select() {
        let plan = build_query("SELECT * FROM users").unwrap();
        assert_eq!(plan.node_type(), "Project");
    }

    #[test]
    fn select_with_where() {
        let plan = build_query("SELECT id, name FROM users WHERE age > 21").unwrap();
        assert_eq!(plan.node_type(), "Project");

        // Check the filter is in the tree
        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Filter"));
        assert!(output.contains("Scan: users"));
    }

    #[test]
    fn select_with_join() {
        let plan =
            build_query("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id")
                .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Join"));
        assert!(output.contains("INNER"));
    }

    #[test]
    fn select_with_aggregate() {
        let plan =
            build_query("SELECT category, COUNT(*) FROM products GROUP BY category").unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Aggregate"));
    }

    #[test]
    fn select_with_order_limit() {
        let plan = build_query("SELECT * FROM users ORDER BY name LIMIT 10 OFFSET 5").unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Sort"));
        assert!(output.contains("Limit"));
    }

    #[test]
    fn insert_values() {
        let plan =
            build_query("INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25)").unwrap();

        assert_eq!(plan.node_type(), "Insert");
    }

    #[test]
    fn update_statement() {
        let plan = build_query("UPDATE users SET status = 'active' WHERE id = 1").unwrap();

        assert_eq!(plan.node_type(), "Update");
    }

    #[test]
    fn delete_statement() {
        let plan = build_query("DELETE FROM users WHERE id = 1").unwrap();
        assert_eq!(plan.node_type(), "Delete");
    }

    #[test]
    fn subquery() {
        let plan = build_query(
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)",
        )
        .unwrap();

        // The plan should contain the subquery
        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Filter"));
    }

    #[test]
    fn union() {
        let plan = build_query("SELECT id FROM users UNION ALL SELECT id FROM admins").unwrap();

        assert_eq!(plan.node_type(), "SetOp");
    }

    #[test]
    fn expression_building() {
        let mut builder = PlanBuilder::new();

        // Test various expression types
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::column(ast::QualifiedName::simple("age"))),
            op: ast::BinaryOp::Gt,
            right: Box::new(Expr::integer(21)),
        };

        let logical = builder.build_expr(&expr).unwrap();
        assert_eq!(logical.to_string(), "(age > 21)");
    }
}
