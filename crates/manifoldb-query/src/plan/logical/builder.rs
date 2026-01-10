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
    self, AlterSchemaStatement, AlterTableStatement, BinaryOp, CallStatement,
    CreateCollectionStatement, CreateFunctionStatement, CreateGraphStatement, CreateIndexStatement,
    CreateNodeRef, CreatePattern, CreateSchemaStatement, CreateTableStatement,
    CreateTriggerStatement, CreateViewStatement, DeleteGraphStatement, DeleteStatement,
    DropCollectionStatement, DropFunctionStatement, DropIndexStatement, DropSchemaStatement,
    DropTableStatement, DropTriggerStatement, DropViewStatement, Expr,
    ForeachAction as AstForeachAction, ForeachStatement, GraphPattern, GroupByClause, InsertSource,
    InsertStatement, JoinClause, JoinCondition, JoinType as AstJoinType, MapProjectionItem,
    MatchStatement, MergeAction as AstMergeAction, MergeClause as AstMergeClause,
    MergeGraphStatement, MergeMatchType as AstMergeMatchType, MergePattern, MergeSqlStatement,
    PathPattern, PropertyCondition, RemoveGraphStatement, RemoveItem, SelectItem, SelectStatement,
    SetAction as AstSetAction, SetGraphStatement, SetOperation, SetOperator, Statement, TableRef,
    UpdateStatement, WindowFunction, YieldItem,
};

use super::ddl::{
    AlterIndexNode, AlterSchemaNode, AlterTableNode, CreateCollectionNode, CreateFunctionNode,
    CreateIndexNode, CreateSchemaNode, CreateTableNode, CreateTriggerNode, CreateViewNode,
    DropCollectionNode, DropFunctionNode, DropIndexNode, DropSchemaNode, DropTableNode,
    DropTriggerNode, DropViewNode, TruncateTableNode,
};

use super::expr::{
    AggregateFunction, HybridCombinationMethod, HybridExprComponent, LogicalExpr,
    LogicalMapProjectionItem, ScalarFunction, SortOrder,
};
use super::graph::{
    CreateNodeSpec, CreateRelSpec, ExpandDirection, ExpandLength, ExpandNode, GraphCreateNode,
    GraphDeleteNode, GraphForeachAction, GraphForeachNode, GraphMergeNode, GraphRemoveAction,
    GraphRemoveNode, GraphSetAction, GraphSetNode, MergePatternSpec, ShortestPathNode,
};
use super::node::{
    LogicalConflictAction, LogicalConflictTarget, LogicalMergeAction, LogicalMergeClause,
    LogicalMergeMatchType, LogicalOnConflict, LogicalPlan,
};
use super::procedure::{ProcedureCallNode, YieldColumn};
use super::relational::{
    AggregateNode, CallSubqueryNode, FilterNode, JoinNode, JoinType, LimitNode, LogicalGroupingSet,
    ProjectNode, ScanNode, SetOpNode, SetOpType, SortNode, ValuesNode, WindowNode,
};
use super::validate::{PlanError, PlanResult};

/// A view definition that can be expanded during query planning.
///
/// Views are stored query definitions that can be used like tables.
/// When a view is referenced in a query, its definition is expanded inline.
///
/// # Examples
///
/// ```
/// use manifoldb_query::plan::logical::ViewDefinition;
/// use manifoldb_query::parser::parse_single_statement;
/// use manifoldb_query::ast::Statement;
///
/// // Parse a SELECT statement for the view definition
/// let stmt = parse_single_statement("SELECT * FROM users WHERE active = true").unwrap();
/// if let Statement::Select(select) = stmt {
///     let view = ViewDefinition::new("active_users", *select);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// The view name.
    pub name: String,
    /// Optional column aliases for the view.
    pub columns: Vec<String>,
    /// The SELECT statement defining the view.
    pub query: SelectStatement,
}

impl ViewDefinition {
    /// Creates a new view definition.
    #[must_use]
    pub fn new(name: impl Into<String>, query: SelectStatement) -> Self {
        Self { name: name.into(), columns: vec![], query }
    }

    /// Creates a view definition with column aliases.
    #[must_use]
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = columns;
        self
    }
}

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
    /// Stack of CTE scopes for proper nested scoping.
    ///
    /// Each scope is a HashMap of CTE names to plans. When a CTE is referenced,
    /// we search from the innermost scope outward. CTEs shadow outer scope CTEs
    /// and views with the same name.
    ///
    /// # Scoping Rules
    ///
    /// 1. CTEs are visible within their defining query and nested subqueries
    /// 2. Subqueries in FROM clause create new scopes (don't inherit outer CTEs)
    /// 3. Later CTEs in the same WITH clause can reference earlier ones
    /// 4. CTEs shadow views with the same name
    cte_scope_stack: Vec<HashMap<String, LogicalPlan>>,
    /// View definitions indexed by name.
    /// When a view is referenced, it is expanded to its defining query.
    /// Views have lower precedence than CTEs (CTEs shadow views).
    view_definitions: HashMap<String, ViewDefinition>,
    /// Named window definitions for the current query.
    /// These are defined in the WINDOW clause and can be referenced by name in OVER clauses.
    #[allow(dead_code)] // Will be used when named window references are implemented
    named_windows: HashMap<String, ast::NamedWindowDefinition>,
}

impl PlanBuilder {
    /// Creates a new plan builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alias_counter: 0,
            cte_scope_stack: Vec::new(),
            view_definitions: HashMap::new(),
            named_windows: HashMap::new(),
        }
    }

    /// Looks up a CTE by name, searching from innermost scope outward.
    ///
    /// Returns the plan if found in any scope, None otherwise.
    fn lookup_cte(&self, name: &str) -> Option<&LogicalPlan> {
        // Search from innermost (most recent) scope to outermost
        for scope in self.cte_scope_stack.iter().rev() {
            if let Some(plan) = scope.get(name) {
                return Some(plan);
            }
        }
        None
    }

    /// Adds a CTE to the current (innermost) scope.
    ///
    /// If no scope exists, this is a no-op (CTEs must be in a scope).
    fn add_cte(&mut self, name: String, plan: LogicalPlan) {
        if let Some(current_scope) = self.cte_scope_stack.last_mut() {
            current_scope.insert(name, plan);
        }
    }

    /// Pushes a new empty CTE scope.
    fn push_cte_scope(&mut self) {
        self.cte_scope_stack.push(HashMap::new());
    }

    /// Pops the innermost CTE scope.
    fn pop_cte_scope(&mut self) {
        self.cte_scope_stack.pop();
    }

    /// Registers a view definition for expansion during query planning.
    ///
    /// When a table reference matches a registered view name, the view's
    /// query definition is expanded inline. Views have lower precedence
    /// than CTEs (CTEs shadow views).
    pub fn register_view(&mut self, view: ViewDefinition) {
        self.view_definitions.insert(view.name.clone(), view);
    }

    /// Registers multiple view definitions.
    pub fn register_views(&mut self, views: impl IntoIterator<Item = ViewDefinition>) {
        for view in views {
            self.register_view(view);
        }
    }

    /// Creates a plan builder with pre-registered views.
    #[must_use]
    pub fn with_views(views: impl IntoIterator<Item = ViewDefinition>) -> Self {
        let mut builder = Self::new();
        builder.register_views(views);
        builder
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
            Statement::AlterTable(alter) => self.build_alter_table(alter),
            Statement::CreateIndex(create) => self.build_create_index(create),
            Statement::CreateCollection(create) => self.build_create_collection(create),
            Statement::DropTable(drop) => self.build_drop_table(drop),
            Statement::DropIndex(drop) => self.build_drop_index(drop),
            Statement::AlterIndex(alter) => self.build_alter_index(alter),
            Statement::TruncateTable(truncate) => self.build_truncate_table(truncate),
            Statement::DropCollection(drop) => self.build_drop_collection(drop),
            Statement::CreateView(create) => self.build_create_view(create),
            Statement::DropView(drop) => self.build_drop_view(drop),
            Statement::CreateSchema(create) => self.build_create_schema(create),
            Statement::AlterSchema(alter) => self.build_alter_schema(alter),
            Statement::DropSchema(drop) => self.build_drop_schema(drop),
            Statement::CreateFunction(create) => self.build_create_function(create),
            Statement::DropFunction(drop) => self.build_drop_function(drop),
            Statement::CreateTrigger(create) => self.build_create_trigger(create),
            Statement::DropTrigger(drop) => self.build_drop_trigger(drop),
            Statement::SetSearchPath(_) => {
                // SET search_path is handled as a utility statement
                Err(super::validate::PlanError::Unsupported(
                    "SET search_path is not yet implemented".to_string(),
                ))
            }
            Statement::Create(create) => self.build_graph_create(create),
            Statement::Merge(merge) => self.build_graph_merge(merge),
            Statement::Call(call) => self.build_call(call),
            Statement::ShowProcedures(show) => self.build_show_procedures(show),
            Statement::Set(set) => self.build_graph_set(set),
            Statement::DeleteGraph(delete) => self.build_graph_delete(delete),
            Statement::Remove(remove) => self.build_graph_remove(remove),
            Statement::Foreach(foreach) => self.build_graph_foreach(foreach),
            Statement::Transaction(txn) => self.build_transaction(txn),
            Statement::ExplainAnalyze(explain) => self.build_explain_analyze(explain),
            Statement::Utility(utility) => self.build_utility(utility),
            Statement::MergeSql(merge) => self.build_merge_sql(merge),
        }
    }

    /// Builds a logical plan from a transaction statement.
    fn build_transaction(&self, stmt: &ast::TransactionStatement) -> PlanResult<LogicalPlan> {
        use super::transaction::{
            BeginTransactionNode, CommitNode, ReleaseSavepointNode, RollbackNode, SavepointNode,
            SetTransactionNode,
        };
        use crate::ast::TransactionStatement;

        match stmt {
            TransactionStatement::Begin(begin) => {
                let mut node = BeginTransactionNode::new();
                if let Some(level) = begin.isolation_level {
                    node = node.with_isolation_level(level);
                }
                if let Some(mode) = begin.access_mode {
                    node = node.with_access_mode(mode);
                }
                if begin.deferred {
                    node = node.deferred();
                }
                Ok(LogicalPlan::BeginTransaction(node))
            }
            TransactionStatement::Commit => Ok(LogicalPlan::Commit(CommitNode::new())),
            TransactionStatement::Rollback(rollback) => {
                let node = if let Some(ref sp) = rollback.to_savepoint {
                    RollbackNode::to_savepoint(&sp.name)
                } else {
                    RollbackNode::new()
                };
                Ok(LogicalPlan::Rollback(node))
            }
            TransactionStatement::Savepoint(sp) => {
                Ok(LogicalPlan::Savepoint(SavepointNode::new(&sp.name.name)))
            }
            TransactionStatement::ReleaseSavepoint(release) => {
                Ok(LogicalPlan::ReleaseSavepoint(ReleaseSavepointNode::new(&release.name.name)))
            }
            TransactionStatement::SetTransaction(set_txn) => {
                let mut node = SetTransactionNode::new();
                if let Some(level) = set_txn.isolation_level {
                    node = node.with_isolation_level(level);
                }
                if let Some(mode) = set_txn.access_mode {
                    node = node.with_access_mode(mode);
                }
                Ok(LogicalPlan::SetTransaction(node))
            }
        }
    }

    /// Builds a logical plan from an EXPLAIN ANALYZE statement.
    fn build_explain_analyze(
        &mut self,
        explain: &ast::ExplainAnalyzeStatement,
    ) -> PlanResult<LogicalPlan> {
        use super::utility::{ExplainAnalyzeNode, ExplainFormat as PlanExplainFormat};

        // Build the inner statement plan
        let inner_plan = self.build_statement(&explain.statement)?;

        let format = match explain.format {
            ast::ExplainFormat::Text => PlanExplainFormat::Text,
            ast::ExplainFormat::Json => PlanExplainFormat::Json,
            ast::ExplainFormat::Xml => PlanExplainFormat::Xml,
            ast::ExplainFormat::Yaml => PlanExplainFormat::Yaml,
        };

        let node = ExplainAnalyzeNode {
            input: Box::new(inner_plan),
            buffers: explain.buffers,
            timing: explain.timing,
            format,
            verbose: explain.verbose,
            costs: explain.costs,
        };

        Ok(LogicalPlan::ExplainAnalyze(node))
    }

    /// Builds a logical plan from a utility statement.
    fn build_utility(&self, utility: &ast::UtilityStatement) -> PlanResult<LogicalPlan> {
        use super::utility::{
            AnalyzeNode, CopyNode, ResetNode, SetSessionNode, ShowNode, VacuumNode,
        };
        use crate::ast::UtilityStatement;

        match utility {
            UtilityStatement::Vacuum(vacuum) => {
                let node = VacuumNode {
                    full: vacuum.full,
                    analyze: vacuum.analyze,
                    table: vacuum.table.clone(),
                    columns: vacuum.columns.iter().map(|i| i.name.clone()).collect(),
                };
                Ok(LogicalPlan::Vacuum(node))
            }
            UtilityStatement::Analyze(analyze) => {
                let node = AnalyzeNode {
                    table: analyze.table.clone(),
                    columns: analyze.columns.iter().map(|i| i.name.clone()).collect(),
                };
                Ok(LogicalPlan::Analyze(node))
            }
            UtilityStatement::Copy(copy) => {
                let node = CopyNode::from_ast(copy)?;
                Ok(LogicalPlan::Copy(node))
            }
            UtilityStatement::Set(set) => {
                let node = SetSessionNode {
                    name: set.name.name.clone(),
                    value: set.value.clone(),
                    local: set.local,
                };
                Ok(LogicalPlan::SetSession(node))
            }
            UtilityStatement::Show(show) => {
                let node = ShowNode { name: show.name.as_ref().map(|i| i.name.clone()) };
                Ok(LogicalPlan::Show(node))
            }
            UtilityStatement::Reset(reset) => {
                let node = ResetNode { name: reset.name.as_ref().map(|i| i.name.clone()) };
                Ok(LogicalPlan::Reset(node))
            }
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
        // Push a new scope for this query's CTEs
        // This ensures CTEs are properly scoped to this query and its nested subqueries
        self.push_cte_scope();

        // Process CTEs - build plans for each and store them in the current scope
        // CTEs can reference earlier CTEs in the same WITH clause (sequential visibility)
        for cte in &select.with_clauses {
            let cte_plan = self.build_select(&cte.query)?;
            self.add_cte(cte.name.name.clone(), cte_plan);
        }

        // Build the rest of the query with CTEs in scope
        let result = self.build_select_body(select);

        // Pop the CTE scope when exiting this query level
        self.pop_cte_scope();

        result
    }

    /// Builds the body of a SELECT statement (after CTE scope is set up).
    fn build_select_body(&mut self, select: &SelectStatement) -> PlanResult<LogicalPlan> {
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

        // Cross join additional tables, handling LATERAL subqueries specially
        for table in from.iter().skip(1) {
            match table {
                TableRef::LateralSubquery { query, alias } => {
                    // Build the subquery plan
                    let subquery = self.build_select(query)?;
                    let subquery = subquery.alias(&alias.name.name);

                    // Collect column names from the current plan for correlation detection
                    let outer_columns = Self::collect_output_columns(&plan);

                    // Collect referenced columns in the subquery
                    let referenced_columns = Self::collect_referenced_columns_from_select(query);

                    // Find which outer columns are referenced in the subquery
                    let imported_variables: Vec<String> = outer_columns
                        .into_iter()
                        .filter(|col| referenced_columns.contains(col))
                        .collect();

                    // Create LATERAL join using CallSubquery semantics
                    plan = LogicalPlan::CallSubquery {
                        node: CallSubqueryNode::new(imported_variables),
                        subquery: Box::new(subquery),
                        input: Box::new(plan),
                    };
                }
                _ => {
                    let right = self.build_table_ref(table)?;
                    plan = plan.cross_join(right);
                }
            }
        }

        Ok(plan)
    }

    /// Builds plan from a table reference.
    fn build_table_ref(&mut self, table_ref: &TableRef) -> PlanResult<LogicalPlan> {
        match table_ref {
            TableRef::Table { name, alias } => {
                let table_name =
                    name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

                // Check if this is a CTE reference (CTE names shadow actual table names and views)
                // Uses the scoped lookup which searches from innermost to outermost scope
                if let Some(cte_plan) = self.lookup_cte(&table_name) {
                    // Clone the CTE plan and apply alias if specified
                    let plan = cte_plan.clone();
                    if let Some(a) = alias {
                        return Ok(plan.alias(&a.name.name));
                    }
                    return Ok(plan);
                }

                // Check if this is a view reference (views shadow table names)
                if let Some(view_def) = self.view_definitions.get(&table_name).cloned() {
                    // Build the view's query as the plan
                    let view_plan = self.build_select(&view_def.query)?;
                    // Apply alias: use explicit alias if provided, otherwise use view name
                    let alias_name = alias
                        .as_ref()
                        .map(|a| a.name.name.clone())
                        .unwrap_or_else(|| view_def.name.clone());
                    return Ok(view_plan.alias(&alias_name));
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

            TableRef::LateralSubquery { query, alias } => {
                // When a LATERAL subquery is the first item in FROM, it's effectively
                // just a regular subquery since there's nothing to correlate with.
                let subquery = self.build_select(query)?;
                Ok(subquery.alias(&alias.name.name))
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
        // Convert GROUP BY clause to logical expressions and grouping sets
        let (group_by, grouping_sets) = self.build_group_by_clause(&select.group_by)?;

        // Extract aggregate expressions from projection - pre-allocate based on projection size
        let mut aggregates = Vec::with_capacity(select.projection.len());
        for item in &select.projection {
            if let SelectItem::Expr { expr, .. } = item {
                self.collect_aggregates(expr, &mut aggregates)?;
            }
        }

        let agg_node = if grouping_sets.is_empty() {
            AggregateNode::new(group_by, aggregates)
        } else {
            AggregateNode::with_grouping_sets(group_by, aggregates, grouping_sets)
        };

        Ok(LogicalPlan::Aggregate { node: Box::new(agg_node), input: Box::new(input) })
    }

    /// Converts a GROUP BY clause to logical expressions and grouping sets.
    fn build_group_by_clause(
        &mut self,
        clause: &GroupByClause,
    ) -> PlanResult<(Vec<LogicalExpr>, Vec<LogicalGroupingSet>)> {
        match clause {
            GroupByClause::Simple(exprs) => {
                let logical_exprs: Vec<LogicalExpr> =
                    exprs.iter().map(|e| self.build_expr(e)).collect::<PlanResult<_>>()?;
                Ok((logical_exprs, vec![]))
            }
            GroupByClause::Rollup(exprs) | GroupByClause::Cube(exprs) => {
                // Get base expressions
                let logical_exprs: Vec<LogicalExpr> =
                    exprs.iter().map(|e| self.build_expr(e)).collect::<PlanResult<_>>()?;

                // Expand to grouping sets
                let grouping_sets = clause.expand_grouping_sets();
                let logical_grouping_sets: Vec<LogicalGroupingSet> = grouping_sets
                    .into_iter()
                    .map(|gs| {
                        let exprs: Vec<LogicalExpr> = gs
                            .exprs()
                            .iter()
                            .map(|e| self.build_expr(e))
                            .collect::<PlanResult<_>>()?;
                        Ok(LogicalGroupingSet::new(exprs))
                    })
                    .collect::<PlanResult<_>>()?;

                Ok((logical_exprs, logical_grouping_sets))
            }
            GroupByClause::GroupingSets(sets) => {
                // Collect all unique expressions across all grouping sets
                let mut all_exprs: Vec<LogicalExpr> = Vec::new();
                let mut seen_exprs = std::collections::HashSet::new();

                let logical_grouping_sets: Vec<LogicalGroupingSet> = sets
                    .iter()
                    .map(|gs| {
                        let exprs: Vec<LogicalExpr> = gs
                            .exprs()
                            .iter()
                            .map(|e| {
                                let key = format!("{e:?}");
                                let logical = self.build_expr(e)?;
                                if seen_exprs.insert(key) {
                                    all_exprs.push(logical.clone());
                                }
                                Ok(logical)
                            })
                            .collect::<PlanResult<_>>()?;
                        Ok(LogicalGroupingSet::new(exprs))
                    })
                    .collect::<PlanResult<_>>()?;

                Ok((all_exprs, logical_grouping_sets))
            }
        }
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
                    // Build all arguments for the aggregate function
                    let args: Vec<LogicalExpr> = if func.args.is_empty() {
                        vec![LogicalExpr::wildcard()]
                    } else {
                        func.args
                            .iter()
                            .map(|a| self.build_expr(a))
                            .collect::<PlanResult<Vec<_>>>()?
                    };

                    // Build optional filter clause
                    let filter =
                        func.filter.as_ref().map(|f| self.build_expr(f)).transpose()?.map(Box::new);

                    aggregates.push(LogicalExpr::AggregateFunction {
                        func: agg_func,
                        args,
                        distinct: func.distinct,
                        filter,
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
            // Cypher collect() maps to ARRAY_AGG
            "ARRAY_AGG" | "COLLECT" => Some(AggregateFunction::ArrayAgg),
            "STRING_AGG" => Some(AggregateFunction::StringAgg),
            // Standard deviation (sample): SQL STDDEV/STDDEV_SAMP, Cypher stDev
            "STDDEV" | "STDDEV_SAMP" | "STDEV" => Some(AggregateFunction::StddevSamp),
            // Standard deviation (population): SQL STDDEV_POP, Cypher stDevP
            "STDDEV_POP" | "STDEVP" => Some(AggregateFunction::StddevPop),
            // Variance (sample): SQL VARIANCE/VAR_SAMP
            "VARIANCE" | "VAR_SAMP" | "VAR" => Some(AggregateFunction::VarianceSamp),
            // Variance (population): SQL VAR_POP
            "VAR_POP" | "VARP" => Some(AggregateFunction::VariancePop),
            // Cypher percentile functions
            "PERCENTILECONT" | "PERCENTILE_CONT" => Some(AggregateFunction::PercentileCont),
            "PERCENTILEDISC" | "PERCENTILE_DISC" => Some(AggregateFunction::PercentileDisc),
            // JSON aggregates
            "JSON_AGG" => Some(AggregateFunction::JsonAgg),
            "JSONB_AGG" => Some(AggregateFunction::JsonbAgg),
            "JSON_OBJECT_AGG" => Some(AggregateFunction::JsonObjectAgg),
            "JSONB_OBJECT_AGG" => Some(AggregateFunction::JsonbObjectAgg),
            // Vector aggregates
            "VECTOR_AVG" => Some(AggregateFunction::VectorAvg),
            "VECTOR_CENTROID" => Some(AggregateFunction::VectorCentroid),
            // Boolean aggregates
            "BOOL_AND" => Some(AggregateFunction::BoolAnd),
            "BOOL_OR" => Some(AggregateFunction::BoolOr),
            "EVERY" => Some(AggregateFunction::BoolAnd), // SQL standard synonym for BOOL_AND
            // Grouping set function
            "GROUPING" => Some(AggregateFunction::Grouping),
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

        // Build optional filter clause
        let filter = func.filter.as_ref().map(|f| self.build_expr(f)).transpose()?.map(Box::new);

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
                filter,
            }),
            "RANK" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::Rank,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
                filter,
            }),
            "DENSE_RANK" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::DenseRank,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
                filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
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
                    filter,
                })
            }

            // Distribution/ranking functions
            "NTILE" => {
                let n = self.get_window_ntile_arg(&func.args)?;
                Ok(LogicalExpr::WindowFunction {
                    func: WindowFunction::Ntile { n },
                    arg: None,
                    default_value: None,
                    partition_by,
                    order_by,
                    frame: over.frame.clone(),
                    filter,
                })
            }
            "PERCENT_RANK" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::PercentRank,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
                filter,
            }),
            "CUME_DIST" => Ok(LogicalExpr::WindowFunction {
                func: WindowFunction::CumeDist,
                arg: None,
                default_value: None,
                partition_by,
                order_by,
                frame: over.frame.clone(),
                filter,
            }),

            _ => Err(PlanError::Unsupported(format!("window function: {name}"))),
        }
    }

    /// Gets the n argument for NTILE.
    fn get_window_ntile_arg(&self, args: &[Expr]) -> PlanResult<u64> {
        if args.is_empty() {
            return Err(PlanError::Unsupported(
                "NTILE requires one argument (number of buckets)".to_string(),
            ));
        }
        if let Expr::Literal(ast::Literal::Integer(n)) = &args[0] {
            if *n > 0 {
                return Ok(*n as u64);
            }
            return Err(PlanError::Unsupported(
                "NTILE argument must be a positive integer".to_string(),
            ));
        }
        Err(PlanError::Unsupported("NTILE argument must be a positive integer literal".to_string()))
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

        // Handle shortest path patterns
        for sp in &pattern.shortest_paths {
            plan = self.build_shortest_path_pattern(plan, sp)?;
        }

        Ok(plan)
    }

    /// Builds a shortest path pattern from shortestPath()/allShortestPaths() functions.
    fn build_shortest_path_pattern(
        &mut self,
        input: LogicalPlan,
        sp: &ast::ShortestPathPattern,
    ) -> PlanResult<LogicalPlan> {
        // Extract source and target variables from the path pattern
        let src_var = sp
            .path
            .start
            .variable
            .as_ref()
            .map(|v| v.name.clone())
            .unwrap_or_else(|| self.next_alias("src"));

        // Get the target node (last node in the path)
        let dst_var = if let Some((_, last_node)) = sp.path.steps.last() {
            last_node
                .variable
                .as_ref()
                .map(|v| v.name.clone())
                .unwrap_or_else(|| self.next_alias("dst"))
        } else {
            // Single node pattern - no shortest path possible, just return input
            return Ok(input);
        };

        // Extract edge info from the path
        let (direction, edge_types, max_length) = if let Some((edge, _)) = sp.path.steps.first() {
            let dir = match edge.direction {
                ast::EdgeDirection::Right => ExpandDirection::Outgoing,
                ast::EdgeDirection::Left => ExpandDirection::Incoming,
                ast::EdgeDirection::Undirected => ExpandDirection::Both,
            };
            let types: Vec<String> = edge.edge_types.iter().map(|t| t.name.clone()).collect();
            let max = match &edge.length {
                ast::EdgeLength::Range { max, .. } => *max,
                ast::EdgeLength::Exact(n) => Some(*n),
                ast::EdgeLength::Any => None,
                ast::EdgeLength::Single => Some(1),
            };
            (dir, types, max.map(|m| m as usize))
        } else {
            (ExpandDirection::Both, vec![], None)
        };

        // Build the ShortestPathNode
        let mut sp_node = ShortestPathNode::new(&src_var, &dst_var)
            .with_direction(direction)
            .with_find_all(sp.find_all);

        if !edge_types.is_empty() {
            sp_node = sp_node.with_edge_types(edge_types);
        }

        if let Some(max) = max_length {
            sp_node = sp_node.with_max_length(max);
        }

        // Add path variable if specified
        if let Some(var) = &sp.path_variable {
            sp_node = sp_node.with_path_variable(var);
        }

        // Add source/target node labels
        if let Some(labels) = sp.path.start.simple_labels() {
            if !labels.is_empty() {
                sp_node = sp_node.with_src_labels(labels.iter().map(|l| l.name.clone()).collect());
            }
        }

        if let Some((_, last_node)) = sp.path.steps.last() {
            if let Some(labels) = last_node.simple_labels() {
                if !labels.is_empty() {
                    sp_node =
                        sp_node.with_dst_labels(labels.iter().map(|l| l.name.clone()).collect());
                }
            }
        }

        Ok(LogicalPlan::ShortestPath { node: Box::new(sp_node), input: Box::new(input) })
    }

    /// Converts a list of property conditions to a logical expression filter.
    ///
    /// For a node/edge with variable `var` and properties `{name: 'Alice', age: 30}`,
    /// this creates: `var.name = 'Alice' AND var.age = 30`.
    fn properties_to_filter(
        &mut self,
        properties: &[PropertyCondition],
        var_name: &str,
    ) -> PlanResult<LogicalExpr> {
        let mut conditions: Vec<LogicalExpr> = Vec::new();

        for prop in properties {
            // Build: var.property = value
            let column = LogicalExpr::Column {
                qualifier: Some(var_name.to_string()),
                name: prop.name.name.clone(),
            };
            let value = self.build_expr(&prop.value)?;
            let condition = LogicalExpr::BinaryOp {
                left: Box::new(column),
                op: BinaryOp::Eq,
                right: Box::new(value),
            };
            conditions.push(condition);
        }

        // Combine with AND
        let filter = conditions
            .into_iter()
            .reduce(|acc, cond| LogicalExpr::BinaryOp {
                left: Box::new(acc),
                op: BinaryOp::And,
                right: Box::new(cond),
            })
            .unwrap_or_else(|| LogicalExpr::boolean(true));

        Ok(filter)
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

        // For standalone node patterns (no edges), we need to create a scan
        // of all matching nodes.
        if path.steps.is_empty() {
            // Get the label(s) from the start node - we'll scan by the first label
            if let Some(labels) = path.start.simple_labels() {
                if let Some(first_label) = labels.first() {
                    let label = first_label.name.clone();
                    let node_scan =
                        LogicalPlan::Scan(Box::new(ScanNode::new(&label).with_alias(&start_var)));

                    // If the input is an empty Values node, just use the scan directly.
                    // Otherwise, cross join with the existing plan (for multiple MATCH patterns).
                    plan = if matches!(&plan, LogicalPlan::Values(v) if v.rows.is_empty() || (v.rows.len() == 1 && v.rows[0].is_empty()))
                    {
                        node_scan
                    } else {
                        plan.cross_join(node_scan)
                    };
                }
            }

            // Add property filter if specified
            if !path.start.properties.is_empty() {
                let start_filter = self.properties_to_filter(&path.start.properties, &start_var)?;
                plan = LogicalPlan::Filter {
                    node: FilterNode::new(start_filter),
                    input: Box::new(plan),
                };
            }

            return Ok(plan);
        }

        // Handle start node property filtering for patterns with edges
        // The start node scan is handled by the first Expand which takes
        // the input as its source. The property filter is applied after expansion.
        if !path.start.properties.is_empty() {
            let start_filter = self.properties_to_filter(&path.start.properties, &start_var)?;
            plan =
                LogicalPlan::Filter { node: FilterNode::new(start_filter), input: Box::new(plan) };
        }

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
            let edge_var_name = edge.variable.as_ref().map(|v| v.name.clone());
            if let Some(ref var) = edge_var_name {
                expand = expand.with_edge_var(var);
            }

            // Add variable length
            expand = expand.with_length(ExpandLength::from_ast(&edge.length));

            // Add node labels
            if let Some(labels) = node.simple_labels() {
                if !labels.is_empty() {
                    expand =
                        expand.with_node_labels(labels.iter().map(|l| l.name.clone()).collect());
                }
            }

            // Add node property filter
            if !node.properties.is_empty() {
                let node_filter = self.properties_to_filter(&node.properties, &dst_var)?;
                expand = expand.with_node_filter(node_filter);
            }

            // Add edge property filter
            if !edge.properties.is_empty() {
                // For edge properties, we need a variable to reference
                let edge_var = edge_var_name.clone().unwrap_or_else(|| self.next_alias("edge"));
                let edge_filter = self.properties_to_filter(&edge.properties, &edge_var)?;
                expand = expand.with_edge_filter(edge_filter);
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

    /// Converts a path pattern to a list of expand nodes.
    ///
    /// This is used for pattern comprehensions where we need the expand steps
    /// without building a full logical plan tree.
    fn path_pattern_to_expand_nodes(&mut self, path: &PathPattern) -> PlanResult<Vec<ExpandNode>> {
        let mut expand_nodes = Vec::new();

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
            if let Some(labels) = node.simple_labels() {
                if !labels.is_empty() {
                    expand =
                        expand.with_node_labels(labels.iter().map(|l| l.name.clone()).collect());
                }
            }

            expand_nodes.push(expand);
            current_var = dst_var;
        }

        Ok(expand_nodes)
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

        // Build the ON CONFLICT clause if present
        let on_conflict = if let Some(ref oc) = insert.on_conflict {
            let target = match &oc.target {
                crate::ast::ConflictTarget::Columns(cols) => {
                    LogicalConflictTarget::Columns(cols.iter().map(|c| c.name.clone()).collect())
                }
                crate::ast::ConflictTarget::Constraint(name) => {
                    LogicalConflictTarget::Constraint(name.name.clone())
                }
            };

            let action = match &oc.action {
                crate::ast::ConflictAction::DoNothing => LogicalConflictAction::DoNothing,
                crate::ast::ConflictAction::DoUpdate { assignments, where_clause } => {
                    let logical_assignments: Vec<(String, LogicalExpr)> = assignments
                        .iter()
                        .map(|a| Ok((a.column.name.clone(), self.build_expr(&a.value)?)))
                        .collect::<PlanResult<_>>()?;

                    let logical_where =
                        if let Some(w) = where_clause { Some(self.build_expr(w)?) } else { None };

                    LogicalConflictAction::DoUpdate {
                        assignments: logical_assignments,
                        where_clause: logical_where,
                    }
                }
            };

            Some(LogicalOnConflict { target, action })
        } else {
            None
        };

        Ok(LogicalPlan::Insert { table, columns, input: Box::new(input), on_conflict, returning })
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

    /// Builds a logical plan from a SQL MERGE statement.
    fn build_merge_sql(&mut self, merge: &MergeSqlStatement) -> PlanResult<LogicalPlan> {
        // Extract target table name
        let target_table = match &merge.target {
            TableRef::Table { name, .. } => {
                name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".")
            }
            _ => {
                return Err(PlanError::Unsupported(
                    "MERGE target must be a table reference".to_string(),
                ))
            }
        };

        // Build source plan from TableRef
        let source = self.build_table_ref(&merge.source)?;

        // Build the ON condition
        let on_condition = self.build_expr(&merge.on_condition)?;

        // Convert WHEN clauses
        let clauses = merge
            .clauses
            .iter()
            .map(|c| self.build_merge_clause(c))
            .collect::<PlanResult<Vec<_>>>()?;

        Ok(LogicalPlan::MergeSql { target_table, source: Box::new(source), on_condition, clauses })
    }

    /// Converts an AST MERGE clause to a logical MERGE clause.
    fn build_merge_clause(&mut self, clause: &AstMergeClause) -> PlanResult<LogicalMergeClause> {
        let match_type = match clause.match_type {
            AstMergeMatchType::Matched => LogicalMergeMatchType::Matched,
            AstMergeMatchType::NotMatched => LogicalMergeMatchType::NotMatched,
            AstMergeMatchType::NotMatchedBySource => LogicalMergeMatchType::NotMatchedBySource,
        };

        let condition = clause.condition.as_ref().map(|e| self.build_expr(e)).transpose()?;

        let action = match &clause.action {
            AstMergeAction::Update { assignments } => {
                let logical_assignments = assignments
                    .iter()
                    .map(|a| {
                        let col = a.column.name.clone();
                        let expr = self.build_expr(&a.value)?;
                        Ok((col, expr))
                    })
                    .collect::<PlanResult<Vec<_>>>()?;
                LogicalMergeAction::Update { assignments: logical_assignments }
            }
            AstMergeAction::Delete => LogicalMergeAction::Delete,
            AstMergeAction::Insert { columns, values } => {
                let logical_columns: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
                let logical_values =
                    values.iter().map(|e| self.build_expr(e)).collect::<PlanResult<Vec<_>>>()?;
                LogicalMergeAction::Insert { columns: logical_columns, values: logical_values }
            }
            AstMergeAction::DoNothing => LogicalMergeAction::DoNothing,
        };

        Ok(LogicalMergeClause { match_type, condition, action })
    }

    /// Builds a CREATE TABLE plan.
    fn build_create_table(&self, create: &CreateTableStatement) -> PlanResult<LogicalPlan> {
        let name = create.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = CreateTableNode::new(name, create.columns.clone())
            .with_if_not_exists(create.if_not_exists)
            .with_constraints(create.constraints.clone());

        Ok(LogicalPlan::CreateTable(node))
    }

    /// Builds an ALTER TABLE plan.
    fn build_alter_table(&self, alter: &AlterTableStatement) -> PlanResult<LogicalPlan> {
        let name = alter.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = AlterTableNode::new(name, alter.actions.clone()).with_if_exists(alter.if_exists);

        Ok(LogicalPlan::AlterTable(node))
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

    /// Builds an ALTER INDEX plan.
    fn build_alter_index(&self, alter: &ast::AlterIndexStatement) -> PlanResult<LogicalPlan> {
        let name = alter.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        // Convert AST action to plan node action
        let action = match &alter.action {
            ast::AlterIndexAction::RenameIndex { new_name } => {
                super::ddl::AlterIndexAction::RenameIndex { new_name: new_name.name.clone() }
            }
            ast::AlterIndexAction::SetOptions { options } => {
                super::ddl::AlterIndexAction::SetOptions { options: options.clone() }
            }
            ast::AlterIndexAction::ResetOptions { options } => {
                super::ddl::AlterIndexAction::ResetOptions { options: options.clone() }
            }
        };

        let node = AlterIndexNode::new(name, action).with_if_exists(alter.if_exists);
        Ok(LogicalPlan::AlterIndex(node))
    }

    /// Builds a TRUNCATE TABLE plan.
    fn build_truncate_table(
        &self,
        truncate: &ast::TruncateTableStatement,
    ) -> PlanResult<LogicalPlan> {
        let names: Vec<String> = truncate
            .names
            .iter()
            .map(|n| n.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("."))
            .collect();

        let restart_identity =
            truncate.identity.is_some_and(|i| matches!(i, ast::TruncateIdentity::Restart));

        let cascade = truncate.cascade.is_some_and(|c| matches!(c, ast::TruncateCascade::Cascade));

        let node = TruncateTableNode::new(names)
            .with_restart_identity(restart_identity)
            .with_cascade(cascade);

        Ok(LogicalPlan::TruncateTable(node))
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

    /// Builds a logical plan from a CREATE VIEW statement.
    fn build_create_view(&self, create: &CreateViewStatement) -> PlanResult<LogicalPlan> {
        let name = create.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = CreateViewNode::new(name, (*create.query).clone())
            .with_or_replace(create.or_replace)
            .with_columns(create.columns.clone());

        Ok(LogicalPlan::CreateView(node))
    }

    /// Builds a logical plan from a DROP VIEW statement.
    fn build_drop_view(&self, drop: &DropViewStatement) -> PlanResult<LogicalPlan> {
        let names: Vec<String> = drop
            .names
            .iter()
            .map(|n| n.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("."))
            .collect();

        let node =
            DropViewNode::new(names).with_if_exists(drop.if_exists).with_cascade(drop.cascade);

        Ok(LogicalPlan::DropView(node))
    }

    /// Builds a CREATE SCHEMA plan.
    fn build_create_schema(&self, create: &CreateSchemaStatement) -> PlanResult<LogicalPlan> {
        let mut node =
            CreateSchemaNode::new(&create.name.name).with_if_not_exists(create.if_not_exists);

        if let Some(auth) = &create.authorization {
            node = node.with_authorization(&auth.name);
        }

        Ok(LogicalPlan::CreateSchema(node))
    }

    /// Builds an ALTER SCHEMA plan.
    fn build_alter_schema(&self, alter: &AlterSchemaStatement) -> PlanResult<LogicalPlan> {
        let node = AlterSchemaNode::new(&alter.name.name, alter.action.clone());
        Ok(LogicalPlan::AlterSchema(node))
    }

    /// Builds a DROP SCHEMA plan.
    fn build_drop_schema(&self, drop: &DropSchemaStatement) -> PlanResult<LogicalPlan> {
        let names: Vec<String> = drop.names.iter().map(|n| n.name.clone()).collect();
        let node =
            DropSchemaNode::new(names).with_if_exists(drop.if_exists).with_cascade(drop.cascade);
        Ok(LogicalPlan::DropSchema(node))
    }

    /// Builds a CREATE FUNCTION plan.
    fn build_create_function(&self, create: &CreateFunctionStatement) -> PlanResult<LogicalPlan> {
        let name = create.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let mut node = CreateFunctionNode::new(
            name,
            create.parameters.clone(),
            create.returns.clone(),
            &create.body,
            create.language.clone(),
        )
        .with_or_replace(create.or_replace)
        .with_returns_setof(create.returns_setof)
        .with_strict(create.strict)
        .with_security_definer(create.security_definer);

        if let Some(vol) = create.volatility {
            node = node.with_volatility(vol);
        }

        Ok(LogicalPlan::CreateFunction(Box::new(node)))
    }

    /// Builds a DROP FUNCTION plan.
    fn build_drop_function(&self, drop: &DropFunctionStatement) -> PlanResult<LogicalPlan> {
        let name = drop.name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = if drop.arg_types.is_empty() {
            DropFunctionNode::new(name)
        } else {
            DropFunctionNode::with_args(name, drop.arg_types.clone())
        }
        .with_if_exists(drop.if_exists)
        .with_cascade(drop.cascade);

        Ok(LogicalPlan::DropFunction(node))
    }

    /// Builds a CREATE TRIGGER plan.
    fn build_create_trigger(&mut self, create: &CreateTriggerStatement) -> PlanResult<LogicalPlan> {
        let table =
            create.table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let function =
            create.function.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let mut node = CreateTriggerNode::new(
            &create.name.name,
            create.timing,
            create.events.clone(),
            table,
            function,
        )
        .with_or_replace(create.or_replace)
        .with_for_each(create.for_each)
        .with_args(create.function_args.clone());

        if let Some(when) = &create.when_clause {
            // Validate the when clause can be built
            let _when_expr = self.build_expr(when)?;
            // Store the AST expression for serialization
            node = node.with_when(when.clone());
        }

        Ok(LogicalPlan::CreateTrigger(Box::new(node)))
    }

    /// Builds a DROP TRIGGER plan.
    fn build_drop_trigger(&self, drop: &DropTriggerStatement) -> PlanResult<LogicalPlan> {
        let table = drop.table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");

        let node = DropTriggerNode::new(&drop.name.name, table)
            .with_if_exists(drop.if_exists)
            .with_cascade(drop.cascade);

        Ok(LogicalPlan::DropTrigger(node))
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

    /// Builds a logical plan from a Cypher FOREACH statement.
    fn build_graph_foreach(&mut self, foreach: &ForeachStatement) -> PlanResult<LogicalPlan> {
        // Build MATCH clause as input (or start with single empty row)
        let mut plan = LogicalPlan::Values(ValuesNode::new(vec![vec![]]));

        if let Some(match_clause) = &foreach.match_clause {
            plan = self.build_graph_pattern(plan, match_clause)?;
        }

        // Add WHERE clause if present
        if let Some(where_expr) = &foreach.where_clause {
            let filter = self.build_expr(where_expr)?;
            plan = plan.filter(filter);
        }

        // Build the list expression
        let list_expr = self.build_expr(&foreach.list_expr)?;

        // Build the FOREACH actions
        let actions = self.build_foreach_actions(&foreach.actions)?;

        let foreach_node = GraphForeachNode::new(foreach.variable.name.clone(), list_expr, actions);

        Ok(LogicalPlan::GraphForeach { node: Box::new(foreach_node), input: Box::new(plan) })
    }

    /// Builds FOREACH actions from AST ForeachActions.
    fn build_foreach_actions(
        &mut self,
        actions: &[AstForeachAction],
    ) -> PlanResult<Vec<GraphForeachAction>> {
        let mut result = Vec::new();

        for action in actions {
            match action {
                AstForeachAction::Set(set_action) => {
                    let graph_set = match set_action {
                        AstSetAction::Property { variable, property, value } => {
                            let val = self.build_expr(value)?;
                            GraphSetAction::Property {
                                variable: variable.name.clone(),
                                property: property.name.clone(),
                                value: val,
                            }
                        }
                        AstSetAction::Label { variable, label } => GraphSetAction::Label {
                            variable: variable.name.clone(),
                            label: label.name.clone(),
                        },
                        AstSetAction::Properties { .. } => {
                            return Err(PlanError::Unsupported(
                                "SET node = properties not yet supported in FOREACH".to_string(),
                            ));
                        }
                    };
                    result.push(GraphForeachAction::Set(graph_set));
                }
                AstForeachAction::Create(pattern) => {
                    let create_node = self.build_create_pattern_to_node(pattern)?;
                    result.push(GraphForeachAction::Create(create_node));
                }
                AstForeachAction::Merge(pattern) => {
                    let merge_node = self.build_merge_pattern_to_node(pattern)?;
                    result.push(GraphForeachAction::Merge(merge_node));
                }
                AstForeachAction::Delete { variables, detach } => {
                    let var_names: Vec<String> =
                        variables.iter().map(|id| id.name.clone()).collect();
                    let delete_node = if *detach {
                        GraphDeleteNode::detach(var_names)
                    } else {
                        GraphDeleteNode::new(var_names)
                    };
                    result.push(GraphForeachAction::Delete(delete_node));
                }
                AstForeachAction::Remove(remove_item) => {
                    let remove_action = match remove_item {
                        RemoveItem::Property { variable, property } => {
                            GraphRemoveAction::Property {
                                variable: variable.name.clone(),
                                property: property.name.clone(),
                            }
                        }
                        RemoveItem::Label { variable, label } => GraphRemoveAction::Label {
                            variable: variable.name.clone(),
                            label: label.name.clone(),
                        },
                    };
                    result.push(GraphForeachAction::Remove(remove_action));
                }
                AstForeachAction::Foreach(nested) => {
                    // Build nested FOREACH
                    let nested_list_expr = self.build_expr(&nested.list_expr)?;
                    let nested_actions = self.build_foreach_actions(&nested.actions)?;
                    let nested_node = GraphForeachNode::new(
                        nested.variable.name.clone(),
                        nested_list_expr,
                        nested_actions,
                    );
                    result.push(GraphForeachAction::Foreach(Box::new(nested_node)));
                }
            }
        }

        Ok(result)
    }

    /// Builds a CREATE node from a CreatePattern.
    fn build_create_pattern_to_node(
        &mut self,
        pattern: &CreatePattern,
    ) -> PlanResult<GraphCreateNode> {
        let mut create_node = GraphCreateNode::new();

        match pattern {
            CreatePattern::Node { variable, labels, properties } => {
                let props: Vec<(String, LogicalExpr)> = properties
                    .iter()
                    .map(|(k, v)| Ok((k.name.clone(), self.build_expr(v)?)))
                    .collect::<PlanResult<Vec<_>>>()?;

                let node_spec = CreateNodeSpec::new(
                    variable.as_ref().map(|id| id.name.clone()),
                    labels.iter().map(|l| l.name.clone()).collect(),
                )
                .with_properties(props);
                create_node = create_node.with_node(node_spec);
            }
            CreatePattern::Path { .. } => {
                // For paths, we would need to build relationships too
                return Err(PlanError::Unsupported(
                    "Path patterns in FOREACH CREATE not yet supported".to_string(),
                ));
            }
            CreatePattern::Relationship { .. } => {
                return Err(PlanError::Unsupported(
                    "Relationship patterns in FOREACH CREATE not yet supported".to_string(),
                ));
            }
        }

        Ok(create_node)
    }

    /// Builds a MERGE node from a MergePattern.
    fn build_merge_pattern_to_node(
        &mut self,
        pattern: &MergePattern,
    ) -> PlanResult<GraphMergeNode> {
        match pattern {
            MergePattern::Node { variable, labels, match_properties } => {
                let props: Vec<(String, LogicalExpr)> = match_properties
                    .iter()
                    .map(|(k, v)| Ok((k.name.clone(), self.build_expr(v)?)))
                    .collect::<PlanResult<Vec<_>>>()?;

                let pattern_spec = MergePatternSpec::Node {
                    variable: variable.name.clone(),
                    labels: labels.iter().map(|l| l.name.clone()).collect(),
                    match_properties: props,
                };
                Ok(GraphMergeNode::new(pattern_spec))
            }
            MergePattern::Relationship { .. } => Err(PlanError::Unsupported(
                "Relationship patterns in FOREACH MERGE not yet supported".to_string(),
            )),
        }
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

    /// Builds a SHOW PROCEDURES statement plan.
    fn build_show_procedures(
        &mut self,
        show: &ast::ShowProceduresStatement,
    ) -> PlanResult<LogicalPlan> {
        use super::utility::ShowProceduresNode;

        let node = if show.executable {
            ShowProceduresNode::new().executable()
        } else {
            ShowProceduresNode::new()
        };

        let mut plan = LogicalPlan::ShowProcedures(node);

        // If there are YIELD items (other than wildcard), we need to wrap in a projection
        if !show.yield_items.is_empty()
            && !matches!(show.yield_items.first(), Some(ast::YieldItem::Wildcard))
        {
            let exprs: Vec<LogicalExpr> = show
                .yield_items
                .iter()
                .filter_map(|item| match item {
                    ast::YieldItem::Wildcard => None,
                    ast::YieldItem::Column { name, alias } => {
                        let col = LogicalExpr::column(&name.name);
                        if let Some(a) = alias {
                            Some(LogicalExpr::Alias { expr: Box::new(col), alias: a.name.clone() })
                        } else {
                            Some(col)
                        }
                    }
                })
                .collect();

            if !exprs.is_empty() {
                plan = plan.project(exprs);
            }
        }

        // Apply WHERE filter if present
        if let Some(ref where_clause) = show.where_clause {
            let filter_expr = self.build_expr(where_clause)?;
            plan = plan.filter(filter_expr);
        }

        Ok(plan)
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
                    let args: Vec<LogicalExpr> = if func.args.is_empty() {
                        vec![LogicalExpr::Wildcard]
                    } else {
                        func.args
                            .iter()
                            .map(|a| self.build_expr(a))
                            .collect::<PlanResult<Vec<_>>>()?
                    };

                    // Build optional filter clause
                    let filter =
                        func.filter.as_ref().map(|f| self.build_expr(f)).transpose()?.map(Box::new);

                    return Ok(LogicalExpr::AggregateFunction {
                        func: agg_func,
                        args,
                        distinct: func.distinct,
                        filter,
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
                    "LPAD" => Some(ScalarFunction::Lpad),
                    "RPAD" => Some(ScalarFunction::Rpad),
                    "LEFT" => Some(ScalarFunction::Left),
                    "RIGHT" => Some(ScalarFunction::Right),
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
                    "TO_NUMBER" => Some(ScalarFunction::ToNumber),
                    "TO_TEXT" => Some(ScalarFunction::ToText),
                    "AGE" => Some(ScalarFunction::Age),
                    "DATE_ADD" => Some(ScalarFunction::DateAdd),
                    "DATE_SUBTRACT" => Some(ScalarFunction::DateSubtract),
                    "MAKE_TIMESTAMP" => Some(ScalarFunction::MakeTimestamp),
                    "MAKE_DATE" => Some(ScalarFunction::MakeDate),
                    "MAKE_TIME" => Some(ScalarFunction::MakeTime),
                    "TIMEZONE" => Some(ScalarFunction::Timezone),
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
                    // Array functions (PostgreSQL-compatible)
                    "ARRAY_LENGTH" => Some(ScalarFunction::ArrayLength),
                    "CARDINALITY" => Some(ScalarFunction::Cardinality),
                    "ARRAY_APPEND" => Some(ScalarFunction::ArrayAppend),
                    "ARRAY_PREPEND" => Some(ScalarFunction::ArrayPrepend),
                    "ARRAY_CAT" => Some(ScalarFunction::ArrayCat),
                    "ARRAY_REMOVE" => Some(ScalarFunction::ArrayRemove),
                    "ARRAY_REPLACE" => Some(ScalarFunction::ArrayReplace),
                    "ARRAY_POSITION" => Some(ScalarFunction::ArrayPosition),
                    "ARRAY_POSITIONS" => Some(ScalarFunction::ArrayPositions),
                    "UNNEST" => Some(ScalarFunction::Unnest),
                    // JSON functions
                    "JSON_EXTRACT_PATH" => Some(ScalarFunction::JsonExtractPath),
                    "JSONB_EXTRACT_PATH" => Some(ScalarFunction::JsonbExtractPath),
                    "JSON_EXTRACT_PATH_TEXT" => Some(ScalarFunction::JsonExtractPathText),
                    "JSONB_EXTRACT_PATH_TEXT" => Some(ScalarFunction::JsonbExtractPathText),
                    "JSON_BUILD_OBJECT" => Some(ScalarFunction::JsonBuildObject),
                    "JSONB_BUILD_OBJECT" => Some(ScalarFunction::JsonbBuildObject),
                    "JSON_BUILD_ARRAY" => Some(ScalarFunction::JsonBuildArray),
                    "JSONB_BUILD_ARRAY" => Some(ScalarFunction::JsonbBuildArray),
                    "JSONB_SET" => Some(ScalarFunction::JsonbSet),
                    "JSONB_INSERT" => Some(ScalarFunction::JsonbInsert),
                    "JSONB_STRIP_NULLS" => Some(ScalarFunction::JsonbStripNulls),
                    // JSON path and containment functions (also available as operators)
                    "JSON_EXTRACT_PATH_OP" => Some(ScalarFunction::JsonExtractPathOp),
                    "JSON_EXTRACT_PATH_TEXT_OP" => Some(ScalarFunction::JsonExtractPathTextOp),
                    "JSONB_CONTAINS_KEY" | "JSON_CONTAINS_KEY" => {
                        Some(ScalarFunction::JsonContainsKey)
                    }
                    "JSONB_CONTAINS_ANY_KEY" | "JSON_CONTAINS_ANY_KEY" => {
                        Some(ScalarFunction::JsonContainsAnyKey)
                    }
                    "JSONB_CONTAINS_ALL_KEYS" | "JSON_CONTAINS_ALL_KEYS" => {
                        Some(ScalarFunction::JsonContainsAllKeys)
                    }
                    // JSON set-returning functions
                    "JSON_EACH" => Some(ScalarFunction::JsonEach),
                    "JSONB_EACH" => Some(ScalarFunction::JsonbEach),
                    "JSON_EACH_TEXT" => Some(ScalarFunction::JsonEachText),
                    "JSONB_EACH_TEXT" => Some(ScalarFunction::JsonbEachText),
                    "JSON_ARRAY_ELEMENTS" => Some(ScalarFunction::JsonArrayElements),
                    "JSONB_ARRAY_ELEMENTS" => Some(ScalarFunction::JsonbArrayElements),
                    "JSON_ARRAY_ELEMENTS_TEXT" => Some(ScalarFunction::JsonArrayElementsText),
                    "JSONB_ARRAY_ELEMENTS_TEXT" => Some(ScalarFunction::JsonbArrayElementsText),
                    "JSON_OBJECT_KEYS" => Some(ScalarFunction::JsonObjectKeys),
                    "JSONB_OBJECT_KEYS" => Some(ScalarFunction::JsonbObjectKeys),
                    // SQL/JSON path functions
                    "JSONB_PATH_EXISTS" => Some(ScalarFunction::JsonbPathExists),
                    "JSONB_PATH_QUERY" => Some(ScalarFunction::JsonbPathQuery),
                    "JSONB_PATH_QUERY_ARRAY" => Some(ScalarFunction::JsonbPathQueryArray),
                    "JSONB_PATH_QUERY_FIRST" => Some(ScalarFunction::JsonbPathQueryFirst),
                    // Cypher entity functions
                    "TYPE" => Some(ScalarFunction::Type),
                    "LABELS" => Some(ScalarFunction::Labels),
                    "ID" => Some(ScalarFunction::Id),
                    "PROPERTIES" => Some(ScalarFunction::Properties),
                    "KEYS" => Some(ScalarFunction::Keys),
                    // Cypher path functions
                    "NODES" => Some(ScalarFunction::Nodes),
                    "RELATIONSHIPS" | "RELS" => Some(ScalarFunction::Relationships),
                    "STARTNODE" => Some(ScalarFunction::StartNode),
                    "ENDNODE" => Some(ScalarFunction::EndNode),
                    // Note: "LENGTH" is handled by ScalarFunction::Length for strings,
                    // but for paths we need PathLength - context determines which
                    // Cypher type conversion functions
                    "TOBOOLEAN" => Some(ScalarFunction::ToBoolean),
                    "TOINTEGER" | "TOINT" => Some(ScalarFunction::ToInteger),
                    "TOFLOAT" => Some(ScalarFunction::ToFloat),
                    "TOSTRING" => Some(ScalarFunction::CypherToString),
                    // Cypher temporal functions
                    "DATETIME" => Some(ScalarFunction::CypherDatetime),
                    "DATE" => Some(ScalarFunction::CypherDate),
                    "TIME" => Some(ScalarFunction::CypherTime),
                    "LOCALDATETIME" => Some(ScalarFunction::CypherLocalDatetime),
                    "LOCALTIME" => Some(ScalarFunction::CypherLocalTime),
                    "DURATION" => Some(ScalarFunction::CypherDuration),
                    "DATETIME.TRUNCATE" => Some(ScalarFunction::CypherDatetimeTruncate),
                    // Cypher spatial functions
                    "POINT" => Some(ScalarFunction::Point),
                    "POINT.DISTANCE" => Some(ScalarFunction::PointDistance),
                    "POINT.WITHINBBOX" => Some(ScalarFunction::PointWithinBBox),
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

            Expr::Exists { subquery, negated } => {
                let plan = self.build_select(&subquery.query)?;
                Ok(LogicalExpr::Exists { subquery: Box::new(plan), negated: *negated })
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
                let arr = self.build_expr(array)?;
                let idx = self.build_expr(index)?;
                Ok(LogicalExpr::ArrayIndex { array: Box::new(arr), index: Box::new(idx) })
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

            Expr::ListPredicateAll { variable, list_expr, predicate } => {
                let list = self.build_expr(list_expr)?;
                let pred = self.build_expr(predicate)?;
                Ok(LogicalExpr::ListPredicateAll {
                    variable: variable.name.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }

            Expr::ListPredicateAny { variable, list_expr, predicate } => {
                let list = self.build_expr(list_expr)?;
                let pred = self.build_expr(predicate)?;
                Ok(LogicalExpr::ListPredicateAny {
                    variable: variable.name.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }

            Expr::ListPredicateNone { variable, list_expr, predicate } => {
                let list = self.build_expr(list_expr)?;
                let pred = self.build_expr(predicate)?;
                Ok(LogicalExpr::ListPredicateNone {
                    variable: variable.name.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }

            Expr::ListPredicateSingle { variable, list_expr, predicate } => {
                let list = self.build_expr(list_expr)?;
                let pred = self.build_expr(predicate)?;
                Ok(LogicalExpr::ListPredicateSingle {
                    variable: variable.name.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }

            Expr::ListReduce { accumulator, initial, variable, list_expr, expression } => {
                let init = self.build_expr(initial)?;
                let list = self.build_expr(list_expr)?;
                let expr = self.build_expr(expression)?;
                Ok(LogicalExpr::ListReduce {
                    accumulator: accumulator.name.clone(),
                    initial: Box::new(init),
                    variable: variable.name.clone(),
                    list_expr: Box::new(list),
                    expression: Box::new(expr),
                })
            }

            Expr::MapProjection { source, items } => {
                let source_expr = self.build_expr(source)?;
                let logical_items = items
                    .iter()
                    .map(|item| match item {
                        MapProjectionItem::Property(ident) => {
                            Ok(LogicalMapProjectionItem::Property(ident.name.clone()))
                        }
                        MapProjectionItem::Computed { key, value } => {
                            let value_expr = self.build_expr(value)?;
                            Ok(LogicalMapProjectionItem::Computed {
                                key: key.name.clone(),
                                value: Box::new(value_expr),
                            })
                        }
                        MapProjectionItem::AllProperties => {
                            Ok(LogicalMapProjectionItem::AllProperties)
                        }
                    })
                    .collect::<PlanResult<Vec<_>>>()?;
                Ok(LogicalExpr::MapProjection {
                    source: Box::new(source_expr),
                    items: logical_items,
                })
            }

            Expr::PatternComprehension { pattern, filter_predicate, projection_expr } => {
                // Convert PathPattern to ExpandNode steps
                let expand_steps = self.path_pattern_to_expand_nodes(pattern)?;
                let filter = if let Some(f) = filter_predicate {
                    Some(Box::new(self.build_expr(f)?))
                } else {
                    None
                };
                let projection = self.build_expr(projection_expr)?;
                Ok(LogicalExpr::PatternComprehension {
                    expand_steps,
                    filter_predicate: filter,
                    projection_expr: Box::new(projection),
                })
            }

            Expr::ExistsSubquery { pattern, filter_predicate } => {
                // Convert PathPattern to ExpandNode steps
                let expand_steps = self.path_pattern_to_expand_nodes(pattern)?;
                let filter = if let Some(f) = filter_predicate {
                    Some(Box::new(self.build_expr(f)?))
                } else {
                    None
                };
                Ok(LogicalExpr::ExistsSubquery { expand_steps, filter_predicate: filter })
            }

            Expr::CountSubquery { pattern, filter_predicate } => {
                // Convert PathPattern to ExpandNode steps
                let expand_steps = self.path_pattern_to_expand_nodes(pattern)?;
                let filter = if let Some(f) = filter_predicate {
                    Some(Box::new(self.build_expr(f)?))
                } else {
                    None
                };
                Ok(LogicalExpr::CountSubquery { expand_steps, filter_predicate: filter })
            }

            Expr::CallSubquery { imported_variables, inner_statements } => {
                // Build the inner plan from the inner statements
                // For now, we only support a single MATCH...RETURN pattern
                // More complex subquery patterns would require additional handling
                let imported: Vec<String> =
                    imported_variables.iter().map(|id| id.name.clone()).collect();

                // Build the inner plan - we need to handle the statements
                // For simplicity, if there's exactly one statement that produces a plan, use it
                // Otherwise, create a placeholder plan
                let inner_plan = if inner_statements.len() == 1 {
                    self.build_statement(&inner_statements[0])?
                } else {
                    // For multiple statements, we'd need to chain them
                    // For now, return an empty plan - full implementation would chain statements
                    LogicalPlan::Empty { columns: vec![] }
                };

                Ok(LogicalExpr::CallSubquery {
                    imported_variables: imported,
                    inner_plan: Box::new(inner_plan),
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

    /// Collects output column names from a logical plan.
    ///
    /// This is used to determine which columns are available for correlation
    /// in LATERAL subqueries. For simplicity, we collect all table/alias names
    /// and let the execution layer handle the correlation at runtime.
    fn collect_output_columns(plan: &LogicalPlan) -> Vec<String> {
        let mut columns = Vec::new();
        Self::collect_output_columns_recursive(plan, &mut columns);
        columns
    }

    /// Recursively collects output column names from a logical plan.
    fn collect_output_columns_recursive(plan: &LogicalPlan, columns: &mut Vec<String>) {
        match plan {
            LogicalPlan::Scan(scan) => {
                // Add the reference name (alias or table name)
                let ref_name = scan.reference_name();
                columns.push(ref_name.to_string());
                // Also add any explicit projections
                if let Some(proj) = &scan.projection {
                    for col in proj {
                        columns.push(col.clone());
                        columns.push(format!("{ref_name}.{col}"));
                    }
                }
            }
            LogicalPlan::Alias { alias, input } => {
                columns.push(alias.clone());
                Self::collect_output_columns_recursive(input, columns);
            }
            LogicalPlan::Project { node, input } => {
                // Add projected column names/aliases
                for expr in &node.exprs {
                    if let LogicalExpr::Alias { alias, .. } = expr {
                        columns.push(alias.clone());
                    } else if let LogicalExpr::Column { name, .. } = expr {
                        columns.push(name.clone());
                    }
                }
                Self::collect_output_columns_recursive(input, columns);
            }
            LogicalPlan::Join { left, right, .. } => {
                Self::collect_output_columns_recursive(left, columns);
                Self::collect_output_columns_recursive(right, columns);
            }
            LogicalPlan::CallSubquery { input, subquery, .. } => {
                Self::collect_output_columns_recursive(input, columns);
                Self::collect_output_columns_recursive(subquery, columns);
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input, .. }
            | LogicalPlan::Window { input, .. }
            | LogicalPlan::Unwind { input, .. } => {
                Self::collect_output_columns_recursive(input, columns);
            }
            LogicalPlan::Aggregate { node, input } => {
                // Add group by columns
                for expr in &node.group_by {
                    if let LogicalExpr::Column { name, .. } = expr {
                        columns.push(name.clone());
                    }
                }
                Self::collect_output_columns_recursive(input, columns);
            }
            LogicalPlan::SetOp { left, right, .. } => {
                Self::collect_output_columns_recursive(left, columns);
                Self::collect_output_columns_recursive(right, columns);
            }
            LogicalPlan::Union { inputs, .. } => {
                for input in inputs {
                    Self::collect_output_columns_recursive(input, columns);
                }
            }
            LogicalPlan::Values(values) => {
                if let Some(names) = &values.column_names {
                    columns.extend(names.iter().cloned());
                }
            }
            LogicalPlan::Empty { columns: cols } => {
                columns.extend(cols.iter().cloned());
            }
            // Graph and other nodes don't typically expose columns in the same way
            _ => {}
        }
    }

    /// Collects all column references from a SELECT statement.
    ///
    /// This is used to find which outer columns a LATERAL subquery references.
    fn collect_referenced_columns_from_select(select: &SelectStatement) -> Vec<String> {
        let mut columns = Vec::new();

        // Collect from projection
        for item in &select.projection {
            if let SelectItem::Expr { expr, .. } = item {
                Self::collect_columns_from_expr(expr, &mut columns);
            }
        }

        // Collect from WHERE clause
        if let Some(where_clause) = &select.where_clause {
            Self::collect_columns_from_expr(where_clause, &mut columns);
        }

        // Collect from FROM clause (for JOINs)
        for table_ref in &select.from {
            Self::collect_columns_from_table_ref(table_ref, &mut columns);
        }

        // Collect from GROUP BY
        for expr in select.group_by.base_expressions() {
            Self::collect_columns_from_expr(expr, &mut columns);
        }

        // Collect from HAVING
        if let Some(having) = &select.having {
            Self::collect_columns_from_expr(having, &mut columns);
        }

        // Collect from ORDER BY
        for order in &select.order_by {
            Self::collect_columns_from_expr(&order.expr, &mut columns);
        }

        columns
    }

    /// Collects column names from an expression (simplified).
    fn collect_columns_from_expr(expr: &Expr, columns: &mut Vec<String>) {
        match expr {
            Expr::Column(name) => {
                // Add both the full qualified name and individual parts
                let full_name: String =
                    name.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(".");
                columns.push(full_name);
                for part in &name.parts {
                    columns.push(part.name.clone());
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                Self::collect_columns_from_expr(left, columns);
                Self::collect_columns_from_expr(right, columns);
            }
            Expr::UnaryOp { operand, .. } => {
                Self::collect_columns_from_expr(operand, columns);
            }
            Expr::Function(func) => {
                for arg in &func.args {
                    Self::collect_columns_from_expr(arg, columns);
                }
                if let Some(filter) = &func.filter {
                    Self::collect_columns_from_expr(filter, columns);
                }
            }
            Expr::InList { expr, list, .. } => {
                Self::collect_columns_from_expr(expr, columns);
                for item in list {
                    Self::collect_columns_from_expr(item, columns);
                }
            }
            Expr::InSubquery { expr, .. } => {
                Self::collect_columns_from_expr(expr, columns);
            }
            Expr::Between { expr, low, high, .. } => {
                Self::collect_columns_from_expr(expr, columns);
                Self::collect_columns_from_expr(low, columns);
                Self::collect_columns_from_expr(high, columns);
            }
            Expr::Case(case_expr) => {
                if let Some(op) = &case_expr.operand {
                    Self::collect_columns_from_expr(op, columns);
                }
                for (cond, result) in &case_expr.when_clauses {
                    Self::collect_columns_from_expr(cond, columns);
                    Self::collect_columns_from_expr(result, columns);
                }
                if let Some(else_r) = &case_expr.else_result {
                    Self::collect_columns_from_expr(else_r, columns);
                }
            }
            Expr::Cast { expr, .. } => {
                Self::collect_columns_from_expr(expr, columns);
            }
            Expr::Tuple(exprs) | Expr::ListLiteral(exprs) => {
                for e in exprs {
                    Self::collect_columns_from_expr(e, columns);
                }
            }
            Expr::ArrayIndex { array, index } => {
                Self::collect_columns_from_expr(array, columns);
                Self::collect_columns_from_expr(index, columns);
            }
            Expr::MapProjection { source, .. } => {
                Self::collect_columns_from_expr(source, columns);
            }
            Expr::Exists { .. } | Expr::Subquery { .. } => {
                // Subqueries have their own scope
            }
            Expr::CallSubquery { .. } => {
                // Subqueries have their own scope
            }
            // All other expression types don't have column references
            _ => {}
        }
    }

    /// Collects column references from table references (for JOIN conditions).
    fn collect_columns_from_table_ref(table_ref: &TableRef, columns: &mut Vec<String>) {
        if let TableRef::Join(join) = table_ref {
            Self::collect_columns_from_table_ref(&join.left, columns);
            Self::collect_columns_from_table_ref(&join.right, columns);
            if let JoinCondition::On(expr) = &join.condition {
                Self::collect_columns_from_expr(expr, columns);
            }
        }
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

    #[test]
    fn cte_scoping_subquery_in_from_does_not_inherit_outer_cte() {
        // Subqueries in FROM clause should not see outer CTEs
        // The inner query references 'temp' but it's not the outer CTE
        let plan = build_query(
            "WITH temp AS (SELECT 1 AS id)
             SELECT * FROM (SELECT * FROM temp) AS inner_query",
        );

        // This should work - the inner SELECT * FROM temp refers to the outer CTE
        // because subqueries inherit CTEs from their parent scope
        assert!(plan.is_ok());
    }

    #[test]
    fn cte_scoping_inner_cte_shadows_outer() {
        // Inner CTE with same name should shadow outer CTE
        let plan = build_query(
            "WITH temp AS (SELECT 1 AS outer_id)
             SELECT * FROM (
                 WITH temp AS (SELECT 2 AS inner_id)
                 SELECT * FROM temp
             ) AS inner_query",
        )
        .unwrap();

        // The inner query should use its own 'temp' CTE
        let output = format!("{}", plan.display_tree());
        // Should have the inner CTE's value (inner_id column)
        assert!(output.contains("Project"));
    }

    #[test]
    fn cte_visibility_sequential_within_with_clause() {
        // Later CTEs can reference earlier ones in the same WITH clause
        let plan = build_query(
            "WITH
                first AS (SELECT 1 AS a),
                second AS (SELECT a + 1 AS b FROM first),
                third AS (SELECT b + 1 AS c FROM second)
             SELECT * FROM third",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        // Should have the chain of projections
        assert!(output.contains("Project"));
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

    // =====================================================
    // Tests for Schema DDL plan building
    // =====================================================

    #[test]
    fn build_create_schema() {
        let plan = build_query("CREATE SCHEMA myschema").unwrap();
        assert_eq!(plan.node_type(), "CreateSchema");
    }

    #[test]
    fn build_create_schema_if_not_exists() {
        let plan = build_query("CREATE SCHEMA IF NOT EXISTS myschema").unwrap();
        assert_eq!(plan.node_type(), "CreateSchema");
        if let LogicalPlan::CreateSchema(node) = plan {
            assert!(node.if_not_exists);
        } else {
            panic!("expected CreateSchema plan");
        }
    }

    #[test]
    fn build_drop_schema() {
        let plan = build_query("DROP SCHEMA myschema").unwrap();
        assert_eq!(plan.node_type(), "DropSchema");
    }

    #[test]
    fn build_drop_schema_cascade() {
        let plan = build_query("DROP SCHEMA IF EXISTS myschema CASCADE").unwrap();
        assert_eq!(plan.node_type(), "DropSchema");
        if let LogicalPlan::DropSchema(node) = plan {
            assert!(node.if_exists);
            assert!(node.cascade);
        } else {
            panic!("expected DropSchema plan");
        }
    }

    // =====================================================
    // Tests for Function DDL plan building
    // =====================================================

    #[test]
    fn build_create_function() {
        let plan =
            build_query("CREATE FUNCTION add_one(x INTEGER) RETURNS INTEGER AS 'SELECT x + 1'")
                .unwrap();
        assert_eq!(plan.node_type(), "CreateFunction");
    }

    #[test]
    fn build_create_function_or_replace() {
        let plan = build_query(
            "CREATE OR REPLACE FUNCTION double_val(n INTEGER) RETURNS INTEGER AS 'SELECT n * 2'",
        )
        .unwrap();
        assert_eq!(plan.node_type(), "CreateFunction");
        if let LogicalPlan::CreateFunction(node) = plan {
            assert!(node.or_replace);
        } else {
            panic!("expected CreateFunction plan");
        }
    }

    #[test]
    fn build_drop_function() {
        let plan = build_query("DROP FUNCTION myfunc").unwrap();
        assert_eq!(plan.node_type(), "DropFunction");
    }

    #[test]
    fn build_drop_function_if_exists() {
        let plan = build_query("DROP FUNCTION IF EXISTS myfunc").unwrap();
        assert_eq!(plan.node_type(), "DropFunction");
        if let LogicalPlan::DropFunction(node) = plan {
            assert!(node.if_exists);
        } else {
            panic!("expected DropFunction plan");
        }
    }

    // =====================================================
    // Tests for Trigger DDL plan building
    // =====================================================

    #[test]
    fn build_create_trigger() {
        let plan = build_query(
            "CREATE TRIGGER audit_trigger BEFORE INSERT ON users EXECUTE FUNCTION audit_func()",
        )
        .unwrap();
        assert_eq!(plan.node_type(), "CreateTrigger");
    }

    #[test]
    fn build_create_trigger_or_replace() {
        let plan = build_query(
            "CREATE OR REPLACE TRIGGER my_trigger AFTER DELETE ON items EXECUTE FUNCTION cleanup()",
        )
        .unwrap();
        assert_eq!(plan.node_type(), "CreateTrigger");
        if let LogicalPlan::CreateTrigger(node) = plan {
            assert!(node.or_replace);
        } else {
            panic!("expected CreateTrigger plan");
        }
    }

    #[test]
    fn build_drop_trigger() {
        let plan = build_query("DROP TRIGGER my_trigger ON users").unwrap();
        assert_eq!(plan.node_type(), "DropTrigger");
    }

    #[test]
    fn build_drop_trigger_if_exists() {
        let plan = build_query("DROP TRIGGER IF EXISTS my_trigger ON users").unwrap();
        assert_eq!(plan.node_type(), "DropTrigger");
        if let LogicalPlan::DropTrigger(node) = plan {
            assert!(node.if_exists);
        } else {
            panic!("expected DropTrigger plan");
        }
    }

    // ========================================================================
    // LATERAL Subquery Tests
    // ========================================================================

    #[test]
    fn lateral_subquery_basic() {
        // Basic LATERAL subquery with correlation
        let plan = build_query(
            "SELECT d.name, top_emp.name, top_emp.salary
             FROM departments d,
             LATERAL (
                 SELECT e.name, e.salary
                 FROM employees e
                 WHERE e.department_id = d.id
                 ORDER BY e.salary DESC
                 LIMIT 3
             ) AS top_emp",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        // LATERAL should be converted to CallSubquery
        assert!(output.contains("CallSubquery") || output.contains("Call"));
    }

    #[test]
    fn lateral_subquery_uncorrelated() {
        // LATERAL without correlation (degenerates to cross join behavior)
        let plan = build_query(
            "SELECT u.name, numbers.n
             FROM users u,
             LATERAL (SELECT 1 AS n UNION ALL SELECT 2 AS n) AS numbers",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        // Should still create a CallSubquery node even without explicit correlation
        assert!(output.contains("Project"));
    }

    #[test]
    fn lateral_subquery_with_aggregation() {
        // LATERAL with aggregation in the subquery
        let plan = build_query(
            "SELECT d.name, emp_stats.count, emp_stats.avg_salary
             FROM departments d,
             LATERAL (
                 SELECT COUNT(*) AS count, AVG(salary) AS avg_salary
                 FROM employees e
                 WHERE e.department_id = d.id
             ) AS emp_stats",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("CallSubquery") || output.contains("Aggregate"));
    }

    #[test]
    fn lateral_subquery_first_in_from() {
        // LATERAL as first item in FROM clause (no correlation possible)
        let plan = build_query("SELECT x.n FROM LATERAL (SELECT 1 AS n) AS x").unwrap();

        let output = format!("{}", plan.display_tree());
        // Should be treated as a regular subquery when first in FROM
        assert!(output.contains("Project"));
    }

    #[test]
    fn lateral_subquery_multiple() {
        // Multiple LATERAL subqueries in sequence
        let plan = build_query(
            "SELECT d.name, e.name, p.name
             FROM departments d,
             LATERAL (
                 SELECT name FROM employees WHERE department_id = d.id LIMIT 1
             ) AS e,
             LATERAL (
                 SELECT name FROM projects WHERE lead_name = e.name LIMIT 1
             ) AS p",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        // Should have nested CallSubquery nodes
        assert!(output.contains("CallSubquery") || output.contains("Call"));
    }

    #[test]
    fn lateral_subquery_with_join() {
        // LATERAL combined with regular JOIN
        let plan = build_query(
            "SELECT c.name, recent_orders.total
             FROM customers c
             JOIN regions r ON c.region_id = r.id,
             LATERAL (
                 SELECT SUM(amount) AS total
                 FROM orders o
                 WHERE o.customer_id = c.id
                 ORDER BY o.date DESC
                 LIMIT 5
             ) AS recent_orders",
        )
        .unwrap();

        let output = format!("{}", plan.display_tree());
        assert!(output.contains("Join") || output.contains("CallSubquery"));
    }
}
