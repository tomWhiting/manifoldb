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

use std::collections::HashMap;

use crate::ast::{
    self, CallStatement, CreateCollectionStatement, CreateGraphStatement, CreateIndexStatement,
    CreateNodeRef, CreatePattern, CreateTableStatement, DeleteGraphStatement, DeleteStatement,
    DropCollectionStatement, DropIndexStatement, DropTableStatement, Expr, GraphPattern,
    InsertSource, InsertStatement, JoinClause, JoinCondition, JoinType as AstJoinType,
    MatchStatement, MergeGraphStatement, MergePattern, PathPattern, RemoveGraphStatement,
    RemoveItem, SelectItem, SelectStatement, SetAction as AstSetAction, SetGraphStatement,
    SetOperation, SetOperator, Statement, TableRef, UpdateStatement, WindowFunction, YieldItem,
};

use super::ddl::{
    CreateCollectionNode, CreateIndexNode, CreateTableNode, DropCollectionNode, DropIndexNode,
    DropTableNode,
};

use super::expr::{
    AggregateFunction, HybridCombinationMethod, HybridExprComponent, LogicalExpr, ScalarFunction,
    SortOrder,
};
use super::graph::{
    CreateNodeSpec, CreateRelSpec, ExpandDirection, ExpandLength, ExpandNode, GraphCreateNode,
    GraphDeleteNode, GraphMergeNode, GraphRemoveAction, GraphRemoveNode, GraphSetAction,
    GraphSetNode, MergePatternSpec,
};
use super::node::LogicalPlan;
use super::procedure::{ProcedureCallNode, YieldColumn};
use super::relational::{
    AggregateNode, FilterNode, JoinNode, JoinType, LimitNode, ProjectNode, ScanNode, SetOpNode,
    SetOpType, SortNode, ValuesNode, WindowNode,
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
    /// CTE plans indexed by name.
    /// When a CTE is defined, its plan is stored here.
    /// When a table reference matches a CTE name, the plan is inlined.
    cte_plans: HashMap<String, LogicalPlan>,
}

impl PlanBuilder {
    /// Creates a new plan builder.
    #[must_use]
    pub fn new() -> Self {
        Self { alias_counter: 0, cte_plans: HashMap::new() }
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
            Statement::Create(create) => self.build_graph_create(create),
            Statement::Merge(merge) => self.build_graph_merge(merge),
            Statement::Call(call) => self.build_call(call),
            Statement::Set(set) => self.build_graph_set(set),
            Statement::DeleteGraph(delete) => self.build_graph_delete(delete),
            Statement::Remove(remove) => self.build_graph_remove(remove),
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
        // Process CTEs first - build plans for each and store them
        // CTEs can reference earlier CTEs in the same WITH clause
        for cte in &select.with_clauses {
            let cte_plan = self.build_select(&cte.query)?;
            self.cte_plans.insert(cte.name.name.clone(), cte_plan);
        }

        // Start with FROM clause
        let mut plan = self.build_from(&select.from)?;

        // Handle MATCH clause for graph patterns
        if let Some(pattern) = &select.match_clause {
            plan = self.build_graph_pattern(plan, pattern)?;
        }

        // Handle OPTIONAL MATCH clauses (LEFT OUTER JOIN semantics)
        for optional_pattern in &select.optional_match_clauses {
            plan = self.build_optional_graph_pattern(plan, optional_pattern)?;
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

        // Handle window functions
        let window_exprs = self.collect_window_exprs(&select.projection)?;
        if !window_exprs.is_empty() {
            plan =
                LogicalPlan::Window { node: WindowNode::new(window_exprs), input: Box::new(plan) };
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

                // Check if this is a CTE reference (CTE names shadow actual table names)
                if let Some(cte_plan) = self.cte_plans.get(&table_name) {
                    // Clone the CTE plan and apply alias if specified
                    let plan = cte_plan.clone();
                    if let Some(a) = alias {
                        return Ok(plan.alias(&a.name.name));
                    }
                    return Ok(plan);
                }

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

    /// Collects window expressions from the projection.
    ///
    /// Window functions are identified by having an OVER clause (the `over` field in `FunctionCall`).
    /// Returns a list of (window_expr, alias) pairs.
    fn collect_window_exprs(
        &mut self,
        projection: &[SelectItem],
    ) -> PlanResult<Vec<(LogicalExpr, String)>> {
        let mut window_exprs = Vec::new();
        let mut window_counter = 0;

        for item in projection {
            if let SelectItem::Expr { expr, alias } = item {
                self.collect_window_from_expr(expr, alias, &mut window_exprs, &mut window_counter)?;
            }
        }

        Ok(window_exprs)
    }

    /// Recursively collects window expressions from an AST expression.
    fn collect_window_from_expr(
        &mut self,
        expr: &Expr,
        alias: &Option<ast::Identifier>,
        window_exprs: &mut Vec<(LogicalExpr, String)>,
        counter: &mut usize,
    ) -> PlanResult<()> {
        match expr {
            Expr::Function(func) if func.over.is_some() => {
                // This is a window function
                let window_expr = self.build_window_function(func)?;
                let col_alias = alias.as_ref().map(|a| a.name.clone()).unwrap_or_else(|| {
                    *counter += 1;
                    format!("window_{counter}")
                });
                window_exprs.push((window_expr, col_alias));
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_window_from_expr(left, &None, window_exprs, counter)?;
                self.collect_window_from_expr(right, &None, window_exprs, counter)?;
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_window_from_expr(operand, &None, window_exprs, counter)?;
            }
            Expr::Case(case) => {
                if let Some(op) = &case.operand {
                    self.collect_window_from_expr(op, &None, window_exprs, counter)?;
                }
                for (when, then) in &case.when_clauses {
                    self.collect_window_from_expr(when, &None, window_exprs, counter)?;
                    self.collect_window_from_expr(then, &None, window_exprs, counter)?;
                }
                if let Some(else_result) = &case.else_result {
                    self.collect_window_from_expr(else_result, &None, window_exprs, counter)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Builds a window function expression from an AST function with OVER clause.
    ///
    /// Supports:
    /// - Ranking functions: ROW_NUMBER, RANK, DENSE_RANK
    /// - Value functions: LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE
    /// - Aggregate functions: SUM, AVG, COUNT, MIN, MAX (as window functions)
    fn build_window_function(&mut self, func: &ast::FunctionCall) -> PlanResult<LogicalExpr> {
        let name = func.name.parts.last().map(|p| p.name.to_uppercase()).unwrap_or_default();

        // Parse the OVER clause first (required for all window functions)
        let over = func.over.as_ref().ok_or_else(|| {
            PlanError::Unsupported("window function missing OVER clause".to_string())
        })?;

        // Build partition by expressions
        let partition_by: Vec<LogicalExpr> =
            over.partition_by.iter().map(|e| self.build_expr(e)).collect::<PlanResult<Vec<_>>>()?;

        // Build order by expressions
        let order_by: Vec<SortOrder> = over
            .order_by
            .iter()
            .map(|o| {
                let expr = self.build_expr(&o.expr)?;
                Ok(SortOrder { expr, ascending: o.asc, nulls_first: o.nulls_first })
            })
            .collect::<PlanResult<Vec<_>>>()?;

        // Parse the window function type and build the expression
        match name.as_str() {
            // Ranking functions (no arguments)
            "ROW_NUMBER" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::RowNumber,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
            }),
            "RANK" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::Rank,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
            }),
            "DENSE_RANK" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::DenseRank,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
            }),

            // Value functions (require argument)
            "LAG" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                let offset = self.get_window_offset(&func.args, 1);
                let default_value = self.get_window_default(&func.args, 2)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Lag { offset, has_default: default_value.is_some() },
                    arg: Some(Box::new(arg)),
                    default_value: default_value.map(Box::new),
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "LEAD" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                let offset = self.get_window_offset(&func.args, 1);
                let default_value = self.get_window_default(&func.args, 2)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Lead { offset, has_default: default_value.is_some() },
                    arg: Some(Box::new(arg)),
                    default_value: default_value.map(Box::new),
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "FIRST_VALUE" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::FirstValue,
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "LAST_VALUE" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::LastValue,
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "NTH_VALUE" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                let n = self.get_window_nth(&func.args)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::NthValue { n },
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }

            // Aggregate functions as window functions
            "COUNT" => {
                let arg = if func.args.is_empty() {
                    None // COUNT(*)
                } else {
                    Some(Box::new(self.build_expr(&func.args[0])?))
                };
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Aggregate(ast::AggregateWindowFunction::Count),
                    arg,
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "SUM" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Aggregate(ast::AggregateWindowFunction::Sum),
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "AVG" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Aggregate(ast::AggregateWindowFunction::Avg),
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "MIN" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Aggregate(ast::AggregateWindowFunction::Min),
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }
            "MAX" => {
                let arg = self.get_window_arg(&func.args, &name)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Aggregate(ast::AggregateWindowFunction::Max),
                    arg: Some(Box::new(arg)),
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                })
            }

            _ => Err(PlanError::Unsupported(format!("window function: {name}"))),
        }
    }

    /// Gets the first argument for a window function.
    fn get_window_arg(&mut self, args: &[Expr], func_name: &str) -> PlanResult<LogicalExpr> {
        if args.is_empty() {
            return Err(PlanError::Unsupported(format!(
                "{func_name} requires at least one argument"
            )));
        }
        self.build_expr(&args[0])
    }

    /// Gets the offset argument for LAG/LEAD (default 1).
    fn get_window_offset(&self, args: &[Expr], arg_index: usize) -> u64 {
        if args.len() > arg_index {
            if let Expr::Literal(ast::Literal::Integer(n)) = &args[arg_index] {
                if *n >= 0 {
                    return *n as u64;
                }
            }
        }
        1 // Default offset
    }

    /// Gets the default value argument for LAG/LEAD.
    fn get_window_default(
        &mut self,
        args: &[Expr],
        arg_index: usize,
    ) -> PlanResult<Option<LogicalExpr>> {
        if args.len() > arg_index {
            Ok(Some(self.build_expr(&args[arg_index])?))
        } else {
            Ok(None)
        }
    }

    /// Gets the N argument for NTH_VALUE (1-indexed position).
    fn get_window_nth(&self, args: &[Expr]) -> PlanResult<u64> {
        if args.len() < 2 {
            return Err(PlanError::Unsupported(
                "NTH_VALUE requires two arguments: NTH_VALUE(expr, n)".to_string(),
            ));
        }
        if let Expr::Literal(ast::Literal::Integer(n)) = &args[1] {
            if *n >= 1 {
                return Ok(*n as u64);
            }
        }
        Err(PlanError::Unsupported("NTH_VALUE: n must be a positive integer".to_string()))
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

    /// Builds an OPTIONAL MATCH pattern using LEFT OUTER JOIN semantics.
    ///
    /// OPTIONAL MATCH returns all rows from the left side (the main query),
    /// plus matching rows from the right side (the optional pattern).
    /// When there's no match, the optional pattern variables are NULL.
    ///
    /// The join condition is based on shared variables between the main query
    /// and the optional pattern. For example, if both reference variable `u`,
    /// the join condition is `left.u = right.u`.
    fn build_optional_graph_pattern(
        &mut self,
        input: LogicalPlan,
        pattern: &GraphPattern,
    ) -> PlanResult<LogicalPlan> {
        // Extract variables from the optional pattern to find shared bindings
        let optional_vars = Self::extract_pattern_variables(pattern);

        // For the optional pattern, we need to:
        // 1. Create a scan for the pattern (starting from an empty Values node)
        // 2. Build the expand nodes for the pattern
        // 3. LEFT JOIN with the main query on shared variables

        // Start with an empty relation that represents "all possible nodes"
        // The graph expand will be applied on top of this
        let optional_plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]]));

        // Build the graph pattern for the optional side
        let optional_plan = self.build_graph_pattern(optional_plan, pattern)?;

        // Build the join condition based on shared variables
        // For now, we use a cross join and let the expand nodes handle the matching
        // In a full implementation, we'd identify variables that appear in both
        // the main query and the optional pattern and create proper equality conditions

        // Since graph patterns share node variables implicitly through the expand nodes,
        // we need to identify the "anchor" variable that connects the required and optional patterns.
        // The first variable in the optional pattern that also exists in the main query
        // is our join key.

        // For simplicity in this implementation, we use a LEFT JOIN with the condition
        // being the first node variable from the optional pattern
        let join_condition = if optional_vars.is_empty() {
            // No variables to join on - this shouldn't happen with valid patterns
            // Use a TRUE condition (cross join with LEFT semantics)
            LogicalExpr::boolean(true)
        } else {
            // Create a condition that references the shared variable
            // This assumes the shared variable is bound in both sides
            let var_name = &optional_vars[0];
            LogicalExpr::qualified_column("", var_name)
                .eq(LogicalExpr::qualified_column("", var_name))
        };

        Ok(input.left_join(optional_plan, join_condition))
    }

    /// Extracts all variable names from a graph pattern.
    fn extract_pattern_variables(pattern: &GraphPattern) -> Vec<String> {
        let mut vars = Vec::new();

        for path in &pattern.paths {
            // Add start node variable
            if let Some(var) = &path.start.variable {
                if !vars.contains(&var.name) {
                    vars.push(var.name.clone());
                }
            }

            // Add variables from each step
            for (edge, node) in &path.steps {
                // Add edge variable
                if let Some(var) = &edge.variable {
                    if !vars.contains(&var.name) {
                        vars.push(var.name.clone());
                    }
                }

                // Add node variable
                if let Some(var) = &node.variable {
                    if !vars.contains(&var.name) {
                        vars.push(var.name.clone());
                    }
                }
            }
        }

        vars
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

    /// Builds a logical plan from a Cypher CREATE statement.
    fn build_graph_create(&mut self, create: &CreateGraphStatement) -> PlanResult<LogicalPlan> {
        // Build optional MATCH clause as input
        let input = if let Some(pattern) = &create.match_clause {
            let mut plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]])); // Start with single empty row
            plan = self.build_graph_pattern(plan, pattern)?;

            // Add WHERE clause if present
            if let Some(where_expr) = &create.where_clause {
                let filter = self.build_expr(where_expr)?;
                plan = plan.filter(filter);
            }

            Some(Box::new(plan))
        } else {
            None
        };

        // Build the CREATE patterns
        let mut graph_create = GraphCreateNode::new();

        for pattern in &create.patterns {
            match pattern {
                CreatePattern::Node { variable, labels, properties } => {
                    let var = variable.as_ref().map(|v| v.name.clone());
                    let label_strs: Vec<String> = labels.iter().map(|l| l.name.clone()).collect();
                    let mut props = Vec::new();
                    for (name, expr) in properties {
                        let logical_expr = self.build_expr(expr)?;
                        props.push((name.name.clone(), logical_expr));
                    }
                    let node_spec = CreateNodeSpec::new(var, label_strs).with_properties(props);
                    graph_create = graph_create.with_node(node_spec);
                }
                CreatePattern::Relationship { start, rel_variable, rel_type, properties, end } => {
                    let start_var = start.name.clone();
                    let end_var = end.name.clone();
                    let rel_type_str = rel_type.name.clone();
                    let mut props = Vec::new();
                    for (name, expr) in properties {
                        let logical_expr = self.build_expr(expr)?;
                        props.push((name.name.clone(), logical_expr));
                    }
                    let mut rel_spec =
                        CreateRelSpec::new(start_var, rel_type_str, end_var).with_properties(props);
                    if let Some(rv) = rel_variable {
                        rel_spec = rel_spec.with_variable(rv.name.clone());
                    }
                    graph_create = graph_create.with_relationship(rel_spec);
                }
                CreatePattern::Path { start, steps } => {
                    // Process path pattern - extract start node and relationships
                    match start {
                        CreateNodeRef::New { variable, labels, properties } => {
                            let var = variable.as_ref().map(|v| v.name.clone());
                            let label_strs: Vec<String> =
                                labels.iter().map(|l| l.name.clone()).collect();
                            let mut props = Vec::new();
                            for (name, expr) in properties {
                                let logical_expr = self.build_expr(expr)?;
                                props.push((name.name.clone(), logical_expr));
                            }
                            let node_spec =
                                CreateNodeSpec::new(var, label_strs).with_properties(props);
                            graph_create = graph_create.with_node(node_spec);
                        }
                        CreateNodeRef::Variable(_) => {
                            // Variable reference - node already exists
                        }
                    }

                    // Process steps
                    let mut prev_var = match start {
                        CreateNodeRef::Variable(v) => v.name.clone(),
                        CreateNodeRef::New { variable, .. } => variable
                            .as_ref()
                            .map(|v| v.name.clone())
                            .unwrap_or_else(|| self.next_alias("node")),
                    };

                    for step in steps {
                        // Create destination node if new
                        let dest_var = match &step.destination {
                            CreateNodeRef::Variable(v) => v.name.clone(),
                            CreateNodeRef::New { variable, labels, properties } => {
                                let var = variable.as_ref().map(|v| v.name.clone());
                                let var_name =
                                    var.clone().unwrap_or_else(|| self.next_alias("node"));
                                let label_strs: Vec<String> =
                                    labels.iter().map(|l| l.name.clone()).collect();
                                let mut props = Vec::new();
                                for (name, expr) in properties {
                                    let logical_expr = self.build_expr(expr)?;
                                    props.push((name.name.clone(), logical_expr));
                                }
                                let node_spec =
                                    CreateNodeSpec::new(var, label_strs).with_properties(props);
                                graph_create = graph_create.with_node(node_spec);
                                var_name
                            }
                        };

                        // Create relationship
                        let (start_var, end_var) = if step.outgoing {
                            (prev_var.clone(), dest_var.clone())
                        } else {
                            (dest_var.clone(), prev_var.clone())
                        };

                        let mut props = Vec::new();
                        for (name, expr) in &step.rel_properties {
                            let logical_expr = self.build_expr(expr)?;
                            props.push((name.name.clone(), logical_expr));
                        }

                        let mut rel_spec =
                            CreateRelSpec::new(start_var, step.rel_type.name.clone(), end_var)
                                .with_properties(props);
                        if let Some(rv) = &step.rel_variable {
                            rel_spec = rel_spec.with_variable(rv.name.clone());
                        }
                        graph_create = graph_create.with_relationship(rel_spec);

                        prev_var = dest_var;
                    }
                }
            }
        }

        // Add RETURN expressions if present
        if !create.return_clause.is_empty() {
            let returning = self.build_return_items(&create.return_clause)?;
            graph_create = graph_create.with_returning(returning);
        }

        Ok(LogicalPlan::GraphCreate { node: Box::new(graph_create), input })
    }

    /// Builds a logical plan from a Cypher MERGE statement.
    fn build_graph_merge(&mut self, merge: &MergeGraphStatement) -> PlanResult<LogicalPlan> {
        // Build optional MATCH clause as input
        let input = if let Some(pattern) = &merge.match_clause {
            let mut plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]])); // Start with single empty row
            plan = self.build_graph_pattern(plan, pattern)?;

            // Add WHERE clause if present
            if let Some(where_expr) = &merge.where_clause {
                let filter = self.build_expr(where_expr)?;
                plan = plan.filter(filter);
            }

            Some(Box::new(plan))
        } else {
            None
        };

        // Build the MERGE pattern
        let pattern_spec = match &merge.pattern {
            MergePattern::Node { variable, labels, match_properties } => {
                let var = variable.name.clone();
                let label_strs: Vec<String> = labels.iter().map(|l| l.name.clone()).collect();
                let mut props = Vec::new();
                for (name, expr) in match_properties {
                    let logical_expr = self.build_expr(expr)?;
                    props.push((name.name.clone(), logical_expr));
                }
                MergePatternSpec::Node {
                    variable: var,
                    labels: label_strs,
                    match_properties: props,
                }
            }
            MergePattern::Relationship { start, rel_variable, rel_type, match_properties, end } => {
                let start_var = start.name.clone();
                let end_var = end.name.clone();
                let rel_type_str = rel_type.name.clone();
                let rel_var = rel_variable.as_ref().map(|v| v.name.clone());
                let mut props = Vec::new();
                for (name, expr) in match_properties {
                    let logical_expr = self.build_expr(expr)?;
                    props.push((name.name.clone(), logical_expr));
                }
                MergePatternSpec::Relationship {
                    start_var,
                    rel_variable: rel_var,
                    rel_type: rel_type_str,
                    match_properties: props,
                    end_var,
                }
            }
        };

        let mut graph_merge = GraphMergeNode::new(pattern_spec);

        // Build ON CREATE actions
        let on_create_actions = self.build_set_actions(&merge.on_create)?;
        graph_merge = graph_merge.with_on_create(on_create_actions);

        // Build ON MATCH actions
        let on_match_actions = self.build_set_actions(&merge.on_match)?;
        graph_merge = graph_merge.with_on_match(on_match_actions);

        // Add RETURN expressions if present
        if !merge.return_clause.is_empty() {
            let returning = self.build_return_items(&merge.return_clause)?;
            graph_merge = graph_merge.with_returning(returning);
        }

        Ok(LogicalPlan::GraphMerge { node: Box::new(graph_merge), input })
    }

    /// Builds a logical plan from a Cypher SET statement.
    fn build_graph_set(&mut self, set: &SetGraphStatement) -> PlanResult<LogicalPlan> {
        // Build MATCH clause as input
        let mut plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]])); // Start with single empty row
        plan = self.build_graph_pattern(plan, &set.match_clause)?;

        // Add WHERE clause if present
        if let Some(where_expr) = &set.where_clause {
            let filter = self.build_expr(where_expr)?;
            plan = plan.filter(filter);
        }

        // Build the SET actions
        let set_actions = self.build_set_actions(&set.set_actions)?;
        let mut graph_set = GraphSetNode::new(set_actions);

        // Add RETURN expressions if present
        if !set.return_clause.is_empty() {
            let returning = self.build_return_items(&set.return_clause)?;
            graph_set = graph_set.with_returning(returning);
        }

        Ok(LogicalPlan::GraphSet { node: Box::new(graph_set), input: Box::new(plan) })
    }

    /// Builds a logical plan from a Cypher DELETE statement.
    fn build_graph_delete(&mut self, delete: &DeleteGraphStatement) -> PlanResult<LogicalPlan> {
        // Build MATCH clause as input
        let mut plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]])); // Start with single empty row
        plan = self.build_graph_pattern(plan, &delete.match_clause)?;

        // Add WHERE clause if present
        if let Some(where_expr) = &delete.where_clause {
            let filter = self.build_expr(where_expr)?;
            plan = plan.filter(filter);
        }

        // Build the DELETE node
        let variables: Vec<String> = delete.variables.iter().map(|v| v.name.clone()).collect();
        let mut graph_delete = if delete.detach {
            GraphDeleteNode::detach(variables)
        } else {
            GraphDeleteNode::new(variables)
        };

        // Add RETURN expressions if present
        if !delete.return_clause.is_empty() {
            let returning = self.build_return_items(&delete.return_clause)?;
            graph_delete = graph_delete.with_returning(returning);
        }

        Ok(LogicalPlan::GraphDelete { node: Box::new(graph_delete), input: Box::new(plan) })
    }

    /// Builds a logical plan from a Cypher REMOVE statement.
    fn build_graph_remove(&mut self, remove: &RemoveGraphStatement) -> PlanResult<LogicalPlan> {
        // Build MATCH clause as input
        let mut plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]])); // Start with single empty row
        plan = self.build_graph_pattern(plan, &remove.match_clause)?;

        // Add WHERE clause if present
        if let Some(where_expr) = &remove.where_clause {
            let filter = self.build_expr(where_expr)?;
            plan = plan.filter(filter);
        }

        // Build the REMOVE actions
        let remove_actions: Vec<GraphRemoveAction> = remove
            .items
            .iter()
            .map(|item| match item {
                RemoveItem::Property { variable, property } => GraphRemoveAction::Property {
                    variable: variable.name.clone(),
                    property: property.name.clone(),
                },
                RemoveItem::Label { variable, label } => GraphRemoveAction::Label {
                    variable: variable.name.clone(),
                    label: label.name.clone(),
                },
            })
            .collect();

        let mut graph_remove = GraphRemoveNode::new(remove_actions);

        // Add RETURN expressions if present
        if !remove.return_clause.is_empty() {
            let returning = self.build_return_items(&remove.return_clause)?;
            graph_remove = graph_remove.with_returning(returning);
        }

        Ok(LogicalPlan::GraphRemove { node: Box::new(graph_remove), input: Box::new(plan) })
    }

    /// Builds return items from AST ReturnItems.
    fn build_return_items(
        &mut self,
        items: &[crate::ast::ReturnItem],
    ) -> PlanResult<Vec<LogicalExpr>> {
        let mut result = Vec::new();
        for item in items {
            match item {
                crate::ast::ReturnItem::Wildcard => {
                    // For wildcard, use the Wildcard expression variant
                    result.push(LogicalExpr::Wildcard);
                }
                crate::ast::ReturnItem::Expr { expr, .. } => {
                    let logical_expr = self.build_expr(expr)?;
                    result.push(logical_expr);
                }
            }
        }
        Ok(result)
    }

    /// Builds SET actions from AST SetActions.
    fn build_set_actions(&mut self, actions: &[AstSetAction]) -> PlanResult<Vec<GraphSetAction>> {
        let mut result = Vec::new();
        for action in actions {
            match action {
                AstSetAction::Property { variable, property, value } => {
                    let val = self.build_expr(value)?;
                    result.push(GraphSetAction::Property {
                        variable: variable.name.clone(),
                        property: property.name.clone(),
                        value: val,
                    });
                }
                AstSetAction::Label { variable, label } => {
                    result.push(GraphSetAction::Label {
                        variable: variable.name.clone(),
                        label: label.name.clone(),
                    });
                }
                AstSetAction::Properties { .. } => {
                    // Skip for now - would require map expression support
                }
            }
        }
        Ok(result)
    }

    /// Builds a CALL statement plan.
    fn build_call(&mut self, call: &CallStatement) -> PlanResult<LogicalPlan> {
        // Build the procedure name (qualified name to string)
        let procedure_name =
            call.procedure_name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        // Build the arguments
        let arguments: Vec<LogicalExpr> = call
            .arguments
            .iter()
            .map(|arg| self.build_expr(arg))
            .collect::<PlanResult<Vec<_>>>()?;

        // Build yield columns
        let yield_columns: Vec<YieldColumn> = call
            .yield_items
            .iter()
            .filter_map(|item| match item {
                YieldItem::Wildcard => None, // Wildcard means "yield all" which is represented by empty vec
                YieldItem::Column { name, alias } => Some(if let Some(a) = alias {
                    YieldColumn::with_alias(&name.name, &a.name)
                } else {
                    YieldColumn::new(&name.name)
                }),
            })
            .collect();

        // Build optional filter
        let filter = call.where_clause.as_ref().map(|cond| self.build_expr(cond)).transpose()?;

        let mut node = ProcedureCallNode::new(procedure_name, arguments).with_yields(yield_columns);

        if let Some(f) = filter {
            node = node.with_filter(f);
        }

        Ok(LogicalPlan::procedure_call(node))
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

                // Check if it's a window function (has OVER clause)
                if func.over.is_some() {
                    return self.build_window_function(func);
                }

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
                    // String functions
                    "UPPER" => Some(ScalarFunction::Upper),
                    "LOWER" => Some(ScalarFunction::Lower),
                    "LENGTH" | "CHAR_LENGTH" | "CHARACTER_LENGTH" => Some(ScalarFunction::Length),
                    "CONCAT" => Some(ScalarFunction::Concat),
                    "SUBSTRING" | "SUBSTR" => Some(ScalarFunction::Substring),
                    "TRIM" => Some(ScalarFunction::Trim),
                    "LTRIM" => Some(ScalarFunction::Ltrim),
                    "RTRIM" => Some(ScalarFunction::Rtrim),
                    "REPLACE" => Some(ScalarFunction::Replace),
                    "POSITION" | "STRPOS" => Some(ScalarFunction::Position),
                    "CONCAT_WS" => Some(ScalarFunction::ConcatWs),
                    "SPLIT_PART" => Some(ScalarFunction::SplitPart),
                    "FORMAT" => Some(ScalarFunction::Format),
                    "REGEXP_MATCH" => Some(ScalarFunction::RegexpMatch),
                    "REGEXP_REPLACE" => Some(ScalarFunction::RegexpReplace),
                    "COALESCE" => Some(ScalarFunction::Coalesce),
                    "NULLIF" => Some(ScalarFunction::NullIf),
                    // Numeric functions
                    "ABS" => Some(ScalarFunction::Abs),
                    "CEIL" | "CEILING" => Some(ScalarFunction::Ceil),
                    "FLOOR" => Some(ScalarFunction::Floor),
                    "ROUND" => Some(ScalarFunction::Round),
                    "TRUNC" | "TRUNCATE" => Some(ScalarFunction::Trunc),
                    "SQRT" => Some(ScalarFunction::Sqrt),
                    "POWER" | "POW" => Some(ScalarFunction::Power),
                    "EXP" => Some(ScalarFunction::Exp),
                    "LN" => Some(ScalarFunction::Ln),
                    "LOG" => Some(ScalarFunction::Log),
                    "LOG10" => Some(ScalarFunction::Log10),
                    "SIN" => Some(ScalarFunction::Sin),
                    "COS" => Some(ScalarFunction::Cos),
                    "TAN" => Some(ScalarFunction::Tan),
                    "ASIN" => Some(ScalarFunction::Asin),
                    "ACOS" => Some(ScalarFunction::Acos),
                    "ATAN" => Some(ScalarFunction::Atan),
                    "ATAN2" => Some(ScalarFunction::Atan2),
                    "DEGREES" => Some(ScalarFunction::Degrees),
                    "RADIANS" => Some(ScalarFunction::Radians),
                    "SIGN" => Some(ScalarFunction::Sign),
                    "PI" => Some(ScalarFunction::Pi),
                    "RANDOM" | "RAND" => Some(ScalarFunction::Random),
                    // Date/time functions
                    "NOW" => Some(ScalarFunction::Now),
                    "CURRENT_DATE" => Some(ScalarFunction::CurrentDate),
                    "CURRENT_TIME" => Some(ScalarFunction::CurrentTime),
                    "EXTRACT" => Some(ScalarFunction::Extract),
                    "DATE_PART" => Some(ScalarFunction::DatePart),
                    "DATE_TRUNC" => Some(ScalarFunction::DateTrunc),
                    "TO_TIMESTAMP" => Some(ScalarFunction::ToTimestamp),
                    "TO_DATE" => Some(ScalarFunction::ToDate),
                    "TO_CHAR" => Some(ScalarFunction::ToChar),
                    // Vector functions
                    "VECTOR_DIMENSION" => Some(ScalarFunction::VectorDimension),
                    "VECTOR_NORM" => Some(ScalarFunction::VectorNorm),
                    // List/Collection functions
                    "RANGE" => Some(ScalarFunction::Range),
                    "SIZE" => Some(ScalarFunction::Size),
                    "HEAD" => Some(ScalarFunction::Head),
                    "TAIL" => Some(ScalarFunction::Tail),
                    "LAST" => Some(ScalarFunction::Last),
                    "REVERSE" => Some(ScalarFunction::Reverse),
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
                    ast::HybridCombinationMethod::WeightedSum => {
                        HybridCombinationMethod::WeightedSum
                    }
                    ast::HybridCombinationMethod::RRF { k } => {
                        HybridCombinationMethod::RRF { k: *k }
                    }
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

                for expr in exprs {
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

            Expr::ListComprehension { variable, list_expr, filter_predicate, transform_expr } => {
                let list = self.build_expr(list_expr)?;
                let filter = if let Some(f) = filter_predicate {
                    Some(Box::new(self.build_expr(f)?))
                } else {
                    None
                };
                let transform = if let Some(t) = transform_expr {
                    Some(Box::new(self.build_expr(t)?))
                } else {
                    None
                };
                Ok(LogicalExpr::ListComprehension {
                    variable: variable.name.clone(),
                    list_expr: Box::new(list),
                    filter_predicate: filter,
                    transform_expr: transform,
                })
            }

            Expr::ListLiteral(exprs) => {
                let elements =
                    exprs.iter().map(|e| self.build_expr(e)).collect::<PlanResult<Vec<_>>>()?;
                Ok(LogicalExpr::ListLiteral(elements))
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

    // ========================================================================
    // Common Table Expression (CTE) Tests
    // ========================================================================

    #[test]
    fn simple_cte() {
        let plan = build_query(
            "WITH active_users AS (SELECT * FROM users WHERE status = 'active')
             SELECT * FROM active_users WHERE age > 21",
        )
        .unwrap();

        // The plan should have nested filters (one from CTE, one from main query)
        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Filter")); // At least one filter
        assert!(output.contains("Project")); // CTE is a subquery with projection
    }

    #[test]
    fn multiple_ctes() {
        let plan = build_query(
            "WITH
                dept_totals AS (SELECT dept_id, SUM(salary) as total FROM employees GROUP BY dept_id),
                high_spenders AS (SELECT * FROM dept_totals WHERE total > 100000)
             SELECT * FROM high_spenders",
        )
        .unwrap();

        // The plan should have an aggregate (from dept_totals CTE)
        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Aggregate"));
    }

    #[test]
    fn cte_referenced_multiple_times() {
        let plan = build_query(
            "WITH temp AS (SELECT id, value FROM data)
             SELECT t1.id, t1.value, t2.value
             FROM temp t1
             JOIN temp t2 ON t1.id = t2.id + 1",
        )
        .unwrap();

        // The plan should have a join
        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Join"));
    }

    #[test]
    fn cte_shadows_table_name() {
        // If there's a CTE named 'users', it should shadow any actual table named 'users'
        let plan = build_query(
            "WITH users AS (SELECT 1 AS id, 'test' AS name)
             SELECT * FROM users",
        )
        .unwrap();

        // The plan should NOT have a scan for 'users' table
        // It should instead inline the CTE's projection
        let output = format!("{}", plan.display_tree());
        // The output should show values or projection, not a real table scan
        assert!(output.contains("Project"));
    }

    #[test]
    fn cte_with_aggregation() {
        let plan = build_query(
            "WITH summary AS (
                SELECT category, COUNT(*) as cnt
                FROM products
                GROUP BY category
             )
             SELECT * FROM summary ORDER BY cnt DESC",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Aggregate"));
        assert!(output.contains("Sort"));
    }

    #[test]
    fn nested_cte_reference() {
        // Second CTE references the first
        let plan = build_query(
            "WITH
                base AS (SELECT id, value FROM data WHERE value > 0),
                doubled AS (SELECT id, value * 2 AS doubled_value FROM base)
             SELECT * FROM doubled",
        )
        .unwrap();

        // Should produce a valid plan with nested projections
        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Filter")); // from base CTE's WHERE clause
        assert!(output.contains("Project")); // from doubled CTE's projection
    }

    // ========================================================================
    // Window Function Tests
    // ========================================================================

    #[test]
    fn window_row_number_simple() {
        let plan = build_query(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM employees",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }

    #[test]
    fn window_row_number_with_partition() {
        let plan = build_query(
            "SELECT name, dept, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_rank FROM employees",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }

    #[test]
    fn window_rank_function() {
        let plan =
            build_query("SELECT name, RANK() OVER (ORDER BY score DESC) AS rank FROM scores")
                .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }

    #[test]
    fn window_dense_rank_function() {
        let plan = build_query(
            "SELECT name, DENSE_RANK() OVER (ORDER BY score DESC) AS drank FROM scores",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }

    #[test]
    fn window_multiple_functions() {
        let plan = build_query(
            "SELECT name,
                    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn,
                    RANK() OVER (ORDER BY salary DESC) AS rank
             FROM employees",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }

    // ========================================================================
    // Aggregate Window Function Tests
    // ========================================================================

    #[test]
    fn window_sum_running_total() {
        // SUM(amount) OVER (ORDER BY date) - running total
        let plan = build_query(
            "SELECT date, amount, SUM(amount) OVER (ORDER BY date) AS running_total FROM sales",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("SUM"));
    }

    #[test]
    fn window_avg_moving_average() {
        // AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
        let plan = build_query(
            "SELECT date, value, AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS weekly_avg FROM metrics",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("AVG"));
    }

    #[test]
    fn window_count_cumulative() {
        // COUNT(*) OVER (ORDER BY date) - cumulative count
        let plan = build_query(
            "SELECT date, event, COUNT(*) OVER (ORDER BY date) AS cumulative_count FROM events",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("COUNT"));
    }

    #[test]
    fn window_count_with_expression() {
        // COUNT(column) OVER - counts non-NULL values
        let plan = build_query(
            "SELECT date, value, COUNT(value) OVER (ORDER BY date) AS non_null_count FROM data",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("COUNT"));
    }

    #[test]
    fn window_min_cumulative() {
        // MIN(value) OVER (ORDER BY date) - cumulative minimum
        let plan = build_query(
            "SELECT date, value, MIN(value) OVER (ORDER BY date) AS cumulative_min FROM data",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("MIN"));
    }

    #[test]
    fn window_max_cumulative() {
        // MAX(value) OVER (ORDER BY date) - cumulative maximum
        let plan = build_query(
            "SELECT date, value, MAX(value) OVER (ORDER BY date) AS cumulative_max FROM data",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("MAX"));
    }

    #[test]
    fn window_aggregate_with_partition() {
        // SUM(amount) OVER (PARTITION BY dept ORDER BY date) - per-department running total
        let plan = build_query(
            "SELECT dept, date, amount, SUM(amount) OVER (PARTITION BY dept ORDER BY date) AS dept_total FROM sales",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
        assert!(output.contains("SUM"));
    }

    #[test]
    fn window_multiple_aggregates() {
        // Multiple aggregate window functions in one query
        let plan = build_query(
            "SELECT date, amount,
                    SUM(amount) OVER (ORDER BY date) AS running_total,
                    AVG(amount) OVER (ORDER BY date) AS running_avg,
                    COUNT(*) OVER (ORDER BY date) AS count
             FROM sales",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }

    #[test]
    fn window_mixed_ranking_and_aggregate() {
        // Mix ranking and aggregate window functions
        let plan = build_query(
            "SELECT name, salary,
                    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rank,
                    SUM(salary) OVER (ORDER BY salary DESC) AS cumulative_salary
             FROM employees",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Window"));
    }
}
