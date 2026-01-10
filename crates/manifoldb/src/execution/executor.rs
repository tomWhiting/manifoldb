//! SQL query and statement execution.
//!
//! This module provides functions to execute SQL queries and statements
//! against the storage layer.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{Entity, Value};
use manifoldb_query::ast::DistanceMetric;
use manifoldb_query::ast::Literal;
use manifoldb_query::ast::Statement;
use manifoldb_query::exec::operators::{
    BruteForceSearchOp, HashAggregateOp, HashJoinOp, NestedLoopJoinOp, SetOpOp, UnionOp, ValuesOp,
};
use manifoldb_query::exec::row::{Row, Schema};
use manifoldb_query::exec::{ExecutionContext, Operator, ResultSet};
use manifoldb_query::plan::logical::ViewDefinition;
use manifoldb_query::plan::logical::{
    AlterIndexNode, AlterTableNode, CreateCollectionNode, CreateIndexNode,
    CreateMaterializedViewNode, CreateTableNode, CreateViewNode, DropCollectionNode, DropIndexNode,
    DropMaterializedViewNode, DropTableNode, DropViewNode, JoinType, LogicalExpr,
    LogicalMergeAction, LogicalMergeClause, LogicalMergeMatchType, ProjectNode,
    RefreshMaterializedViewNode, SetOpNode, TruncateTableNode, UnionNode, ValuesNode,
};
use manifoldb_query::plan::logical::{AnnSearchNode, ExpandNode, PathScanNode, VectorDistanceNode};
use manifoldb_query::plan::physical::{IndexInfo, IndexType, PlannerCatalog};
use manifoldb_query::plan::{LogicalPlan, PhysicalPlan, PhysicalPlanner, PlanBuilder};
use manifoldb_query::procedure::builtins::{
    execute_all_shortest_paths_with_tx, execute_astar_with_tx, execute_betweenness_with_tx,
    execute_bfs_with_tx, execute_closeness_with_tx, execute_connected_components_with_tx,
    execute_cosine_with_tx, execute_degree_with_tx, execute_dfs_with_tx, execute_dijkstra_with_tx,
    execute_eigenvector_with_tx, execute_jaccard_with_tx, execute_label_propagation_with_tx,
    execute_louvain_with_tx, execute_node_similarity_with_tx, execute_overlap_with_tx,
    execute_pagerank_with_tx, execute_shortest_path_with_tx, execute_sssp_with_tx,
    execute_strongly_connected_with_tx,
};
use manifoldb_query::ExtendedParser;
use manifoldb_storage::Transaction;

use crate::prepared::PreparedStatement;

use super::graph_accessor;
use super::StorageScan;
use crate::error::{Error, Result};
use crate::schema::SchemaManager;
use crate::transaction::DatabaseTransaction;

/// Execute a SELECT query and return the result set.
pub fn execute_query<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
) -> Result<ResultSet> {
    execute_query_with_limit(tx, sql, params, 0)
}

/// Execute a SELECT query with a row limit and return the result set.
///
/// # Arguments
///
/// * `tx` - The transaction to execute against
/// * `sql` - The SQL query to execute
/// * `params` - The parameter values
/// * `max_rows_in_memory` - Maximum rows operators can materialize (0 = no limit)
pub fn execute_query_with_limit<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
    max_rows_in_memory: usize,
) -> Result<ResultSet> {
    execute_query_with_catalog(tx, sql, params, max_rows_in_memory, None)
}

/// Execute a SELECT query with a custom planner catalog.
///
/// This allows callers to provide additional index information (e.g., payload indexes)
/// that the query planner can use for index selection.
///
/// # Arguments
///
/// * `tx` - The transaction to execute against
/// * `sql` - The SQL query to execute
/// * `params` - The parameter values
/// * `max_rows_in_memory` - Maximum rows operators can materialize (0 = no limit)
/// * `external_catalog` - Optional catalog with additional index info (merged with schema indexes)
pub fn execute_query_with_catalog<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
    max_rows_in_memory: usize,
    external_catalog: Option<PlannerCatalog>,
) -> Result<ResultSet> {
    // Parse SQL using ExtendedParser to support MATCH syntax
    let stmt = ExtendedParser::parse_single(sql)?;

    // Check if this is an EXPLAIN statement
    let (is_explain, inner_stmt) = match &stmt {
        Statement::Explain(inner) => (true, inner.as_ref()),
        other => (false, other),
    };

    // Build logical plan from the inner statement
    // Register views so they can be expanded during planning
    let mut builder = PlanBuilder::new();
    load_views_into_builder(tx, &mut builder)?;
    let logical_plan =
        builder.build_statement(inner_stmt).map_err(|e| Error::Parse(e.to_string()))?;

    // Build catalog with available indexes from schema
    let mut catalog = build_planner_catalog(tx)?;

    // Merge in external catalog indexes (e.g., payload indexes)
    if let Some(external) = external_catalog {
        catalog = catalog.merge(external);
    }

    // Build physical plan with catalog for index selection
    let planner = PhysicalPlanner::new().with_catalog(catalog);
    let physical_plan = planner.plan(&logical_plan);

    // If EXPLAIN, return the plan as a text result instead of executing
    if is_explain {
        return Ok(build_explain_result(&logical_plan, &physical_plan));
    }

    // Create execution context with parameters and row limit
    let ctx = create_context_with_limit(params, max_rows_in_memory);

    // Execute the plan against storage
    execute_physical_plan(tx, &physical_plan, &logical_plan, &ctx)
}

/// Build a result set containing the EXPLAIN output.
fn build_explain_result(logical: &LogicalPlan, physical: &PhysicalPlan) -> ResultSet {
    let schema = Arc::new(Schema::new(vec!["plan".to_string()]));

    let mut rows = Vec::new();

    // Add logical plan header
    rows.push(Row::new(schema.clone(), vec![Value::from("=== Logical Plan ===")]));

    // Add logical plan tree (convert DisplayTree to String)
    let logical_tree = logical.display_tree().to_string();
    for line in logical_tree.lines() {
        rows.push(Row::new(schema.clone(), vec![Value::from(line)]));
    }

    // Add separator
    rows.push(Row::new(schema.clone(), vec![Value::from("")]));

    // Add physical plan header
    rows.push(Row::new(schema.clone(), vec![Value::from("=== Physical Plan ===")]));

    // Add physical plan tree (convert to String)
    let physical_tree = physical.display_tree().to_string();
    for line in physical_tree.lines() {
        rows.push(Row::new(schema.clone(), vec![Value::from(line)]));
    }

    ResultSet::with_rows(schema, rows)
}

/// Execute a DML statement (INSERT, UPDATE, DELETE) and return the affected row count.
pub fn execute_statement<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
) -> Result<u64> {
    // Parse SQL using ExtendedParser to support MATCH syntax
    let stmt = ExtendedParser::parse_single(sql)?;

    // Build logical plan
    let mut builder = PlanBuilder::new();
    let logical_plan = builder.build_statement(&stmt).map_err(|e| Error::Parse(e.to_string()))?;

    // Create execution context with parameters
    let ctx = create_context(params);

    // Execute based on the statement type
    match &logical_plan {
        LogicalPlan::Insert { table, columns, input, on_conflict, .. } => {
            execute_insert(tx, table, columns, input, on_conflict.as_ref(), &ctx)
        }
        LogicalPlan::Update { table, assignments, source, filter, .. } => {
            execute_update(tx, table, assignments, source, filter, &ctx)
        }
        LogicalPlan::Delete { table, source, filter, .. } => {
            execute_delete(tx, table, source, filter, &ctx)
        }
        LogicalPlan::MergeSql { target_table, source, on_condition, clauses } => {
            execute_merge_sql(tx, target_table, source, on_condition, clauses, &ctx)
        }

        // DDL statements
        LogicalPlan::CreateTable(node) => execute_create_table(tx, node),
        LogicalPlan::AlterTable(node) => execute_alter_table(tx, node),
        LogicalPlan::DropTable(node) => execute_drop_table(tx, node),
        LogicalPlan::CreateIndex(node) => execute_create_index(tx, node),
        LogicalPlan::DropIndex(node) => execute_drop_index(tx, node),
        LogicalPlan::AlterIndex(node) => execute_alter_index(tx, node),
        LogicalPlan::TruncateTable(node) => execute_truncate_table(tx, node),
        LogicalPlan::CreateCollection(node) => execute_create_collection(tx, node),
        LogicalPlan::DropCollection(node) => execute_drop_collection(tx, node),
        LogicalPlan::CreateView(node) => execute_create_view(tx, node, sql),
        LogicalPlan::DropView(node) => execute_drop_view(tx, node),
        LogicalPlan::CreateMaterializedView(node) => {
            execute_create_materialized_view(tx, node, sql)
        }
        LogicalPlan::DropMaterializedView(node) => execute_drop_materialized_view(tx, node),
        LogicalPlan::RefreshMaterializedView(node) => execute_refresh_materialized_view(tx, node),

        _ => {
            // For SELECT, we shouldn't be here but handle gracefully
            Err(Error::Execution("Expected DML or DDL statement".to_string()))
        }
    }
}

/// Execute a Cypher graph DML statement (CREATE, MERGE, DELETE, etc.) and return results.
///
/// This function handles graph mutations that can return results (e.g., CREATE with RETURN).
/// It uses a write transaction and the physical plan execution infrastructure.
///
/// # Arguments
///
/// * `tx` - The mutable transaction to execute against
/// * `sql` - The Cypher statement to execute
/// * `params` - The parameter values
/// * `max_rows_in_memory` - Maximum rows operators can materialize (0 = no limit)
pub fn execute_graph_dml<T: Transaction + Send + Sync + 'static>(
    tx: DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
    max_rows_in_memory: usize,
) -> Result<(ResultSet, DatabaseTransaction<T>)> {
    use super::{DatabaseGraphAccessor, DatabaseGraphMutator};
    use manifoldb_query::exec::graph_accessor::{GraphAccessor, GraphMutator};

    // Parse SQL using ExtendedParser to support Cypher syntax
    let stmt = ExtendedParser::parse_single(sql)?;

    // Build logical plan
    let mut builder = PlanBuilder::new();
    let logical_plan = builder.build_statement(&stmt).map_err(|e| Error::Parse(e.to_string()))?;

    // Verify this is a graph DML operation
    if !is_graph_dml(&logical_plan) {
        return Err(Error::Execution(
            "execute_graph_dml requires a graph DML statement (CREATE, MERGE, etc.)".to_string(),
        ));
    }

    // Build physical plan
    let catalog = build_planner_catalog(&tx)?;
    let planner = PhysicalPlanner::new().with_catalog(catalog);
    let physical_plan = planner.plan(&logical_plan);

    // Create a graph mutator wrapping the transaction
    let mutator = DatabaseGraphMutator::new(tx);

    // Create a graph accessor sharing the same transaction for MATCH + CREATE patterns
    let accessor = DatabaseGraphAccessor::from_arc(mutator.transaction_arc());

    // Create execution context with parameters, row limit, graph mutator, and graph accessor
    let mut ctx = create_context_with_limit(params, max_rows_in_memory);
    ctx = ctx.with_graph_mutator(Arc::new(mutator.clone()) as Arc<dyn GraphMutator>);
    ctx = ctx.with_graph(Arc::new(accessor) as Arc<dyn GraphAccessor>);

    // Execute the physical plan using the query executor
    let result = execute_graph_physical_plan(&physical_plan, &ctx)?;

    // Take the transaction back from the mutator
    let tx = mutator.take_transaction().ok_or_else(|| {
        Error::Execution("Transaction was already taken from mutator".to_string())
    })?;

    Ok((result, tx))
}

/// Execute a graph DML physical plan.
fn execute_graph_physical_plan(
    physical: &PhysicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    use manifoldb_query::exec::build_operator_tree;

    // Build the operator tree from the physical plan
    let mut operator =
        build_operator_tree(physical).map_err(|e| Error::Execution(e.to_string()))?;

    // Open the operator
    operator.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    // Collect all rows
    let mut rows = Vec::new();
    while let Some(row) = operator.next().map_err(|e| Error::Execution(e.to_string()))? {
        rows.push(row);
    }

    // Close the operator
    operator.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(operator.schema(), rows))
}

/// Check if a logical plan is a graph DML operation.
fn is_graph_dml(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::GraphCreate { .. }
            | LogicalPlan::GraphMerge { .. }
            | LogicalPlan::GraphSet { .. }
            | LogicalPlan::GraphDelete { .. }
            | LogicalPlan::GraphRemove { .. }
            | LogicalPlan::GraphForeach { .. }
    )
}

/// Check if a SQL string appears to be a Cypher graph DML statement.
///
/// This is a quick heuristic check that can be used before parsing.
pub fn is_cypher_dml(sql: &str) -> bool {
    let sql_upper = sql.trim().to_uppercase();
    // Cypher CREATE starts with CREATE followed by a pattern (parenthesis)
    // SQL CREATE TABLE/INDEX starts with CREATE TABLE/INDEX
    if sql_upper.starts_with("CREATE") {
        // Check if it's a SQL CREATE statement
        if sql_upper.starts_with("CREATE TABLE")
            || sql_upper.starts_with("CREATE INDEX")
            || sql_upper.starts_with("CREATE COLLECTION")
        {
            return false;
        }
        // If CREATE is followed by ( or :, it's likely Cypher
        let after_create = sql_upper.strip_prefix("CREATE").unwrap_or("").trim();
        if after_create.starts_with('(') {
            return true;
        }
    }
    // MATCH ... CREATE/SET/DELETE/etc is Cypher
    if sql_upper.starts_with("MATCH")
        && (sql_upper.contains("CREATE")
            || sql_upper.contains(" SET ")
            || sql_upper.contains("DELETE")
            || sql_upper.contains("REMOVE")
            || sql_upper.contains("MERGE"))
    {
        return true;
    }
    // Other Cypher DML keywords at the start
    sql_upper.starts_with("MERGE")
        || sql_upper.starts_with("DELETE")
        || sql_upper.starts_with("DETACH DELETE")
        || sql_upper.starts_with("REMOVE")
        || sql_upper.starts_with("SET")
        || sql_upper.starts_with("FOREACH")
}

/// Execute a prepared SELECT query and return the result set.
///
/// This uses the cached logical and physical plans from the prepared statement,
/// avoiding the parsing and planning overhead.
pub fn execute_prepared_query<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    stmt: &PreparedStatement,
    params: &[Value],
) -> Result<ResultSet> {
    // Create execution context with parameters
    let ctx = create_context(params);

    // Execute the plan against storage using cached plans
    execute_physical_plan(tx, stmt.physical_plan(), stmt.logical_plan(), &ctx)
}

/// Execute a prepared DML/DDL statement and return the affected row count.
///
/// This uses the cached logical plan from the prepared statement,
/// avoiding the parsing and planning overhead.
pub fn execute_prepared_statement<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    stmt: &PreparedStatement,
    params: &[Value],
) -> Result<u64> {
    // Create execution context with parameters
    let ctx = create_context(params);

    // Execute based on the statement type
    match stmt.logical_plan() {
        LogicalPlan::Insert { table, columns, input, on_conflict, .. } => {
            execute_insert(tx, table, columns, input, on_conflict.as_ref(), &ctx)
        }
        LogicalPlan::Update { table, assignments, source, filter, .. } => {
            execute_update(tx, table, assignments, source, filter, &ctx)
        }
        LogicalPlan::Delete { table, source, filter, .. } => {
            execute_delete(tx, table, source, filter, &ctx)
        }
        LogicalPlan::MergeSql { target_table, source, on_condition, clauses } => {
            execute_merge_sql(tx, target_table, source, on_condition, clauses, &ctx)
        }

        // DDL statements
        LogicalPlan::CreateTable(node) => execute_create_table(tx, node),
        LogicalPlan::AlterTable(node) => execute_alter_table(tx, node),
        LogicalPlan::DropTable(node) => execute_drop_table(tx, node),
        LogicalPlan::CreateIndex(node) => execute_create_index(tx, node),
        LogicalPlan::DropIndex(node) => execute_drop_index(tx, node),
        LogicalPlan::AlterIndex(node) => execute_alter_index(tx, node),
        LogicalPlan::TruncateTable(node) => execute_truncate_table(tx, node),
        LogicalPlan::CreateCollection(node) => execute_create_collection(tx, node),
        LogicalPlan::DropCollection(node) => execute_drop_collection(tx, node),

        _ => {
            // For SELECT, we shouldn't be here but handle gracefully
            Err(Error::Execution("Expected DML or DDL statement".to_string()))
        }
    }
}

/// Create an execution context with bound parameters.
fn create_context(params: &[Value]) -> ExecutionContext {
    create_context_with_limit(params, 0)
}

fn create_context_with_limit(params: &[Value], max_rows_in_memory: usize) -> ExecutionContext {
    use manifoldb_query::exec::ExecutionConfig;

    let mut param_map = HashMap::new();
    for (i, value) in params.iter().enumerate() {
        param_map.insert((i + 1) as u32, value.clone());
    }

    let config = ExecutionConfig::new().with_max_rows_in_memory(max_rows_in_memory);
    ExecutionContext::with_parameters(param_map).with_config(config)
}

/// Load view definitions from the schema and register them with the plan builder.
///
/// This allows views to be expanded during query planning. Views are stored as
/// SQL strings in the schema, so they must be re-parsed when loaded.
fn load_views_into_builder<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    builder: &mut PlanBuilder,
) -> Result<()> {
    // Get all views from the schema
    let view_names = SchemaManager::list_views(tx).unwrap_or_default();

    for name in view_names {
        if let Ok(Some(schema)) = SchemaManager::get_view(tx, &name) {
            // The view query is stored as a debug representation, which isn't directly parseable.
            // We need to store actual SQL in the schema. For now, try to parse it.
            // If the query_sql is valid SQL, parse and register it.
            if let Ok(stmt) = ExtendedParser::parse_single(&schema.query_sql) {
                if let Statement::Select(select) = stmt {
                    let view_def =
                        ViewDefinition::new(&schema.name, *select).with_columns(schema.columns);
                    builder.register_view(view_def);
                }
            }
        }
    }

    // Note: Materialized views are NOT expanded during planning.
    // Instead, they are resolved as scans and the cached data is returned
    // during execution. See try_execute_materialized_view_scan.

    Ok(())
}

/// Try to execute a scan as a materialized view scan.
///
/// If the table name refers to a materialized view, returns the cached data.
/// Returns Ok(None) if the table is not a materialized view.
fn try_execute_materialized_view_scan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    table_name: &str,
    alias: Option<&str>,
) -> Result<Option<ResultSet>> {
    // Check if this table name is a materialized view
    if !SchemaManager::materialized_view_exists(tx, table_name).unwrap_or(false) {
        return Ok(None);
    }

    // Get the cached data
    let data = SchemaManager::get_materialized_view_data(tx, table_name)
        .map_err(|e| Error::Execution(format!("Failed to get materialized view data: {e}")))?
        .ok_or_else(|| {
            Error::Execution(format!(
                "Materialized view '{}' has not been refreshed yet",
                table_name
            ))
        })?;

    let cached_rows = SchemaManager::get_materialized_view_rows(tx, table_name)
        .map_err(|e| Error::Execution(format!("Failed to get materialized view rows: {e}")))?
        .ok_or_else(|| {
            Error::Execution(format!(
                "Materialized view '{}' has not been refreshed yet",
                table_name
            ))
        })?;

    // Build schema with the result columns (optionally aliased)
    let prefix = alias.unwrap_or(table_name);
    let columns: Vec<String> =
        data.result_columns.iter().map(|c| format!("{}.{}", prefix, c)).collect();
    let schema = Arc::new(Schema::new(columns));

    // Convert cached rows to Row objects
    let rows: Vec<Row> =
        cached_rows.rows.into_iter().map(|values| Row::new(Arc::clone(&schema), values)).collect();

    Ok(Some(ResultSet::with_rows(schema, rows)))
}

/// Try to execute a scan as a materialized view and return as entities.
///
/// If the table name refers to a materialized view, returns synthetic entities.
/// Returns Ok(None) if the table is not a materialized view.
fn try_execute_materialized_view_as_entities<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    table_name: &str,
) -> Result<Option<Vec<Entity>>> {
    // Check if this table name is a materialized view
    if !SchemaManager::materialized_view_exists(tx, table_name).unwrap_or(false) {
        return Ok(None);
    }

    // Get the cached data
    let data = SchemaManager::get_materialized_view_data(tx, table_name)
        .map_err(|e| Error::Execution(format!("Failed to get materialized view data: {e}")))?
        .ok_or_else(|| {
            Error::Execution(format!(
                "Materialized view '{}' has not been refreshed yet",
                table_name
            ))
        })?;

    let cached_rows = SchemaManager::get_materialized_view_rows(tx, table_name)
        .map_err(|e| Error::Execution(format!("Failed to get materialized view rows: {e}")))?
        .ok_or_else(|| {
            Error::Execution(format!(
                "Materialized view '{}' has not been refreshed yet",
                table_name
            ))
        })?;

    // Convert cached rows to synthetic entities
    let entities: Vec<Entity> = cached_rows
        .rows
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            let mut entity = Entity::new(manifoldb_core::EntityId::new(i as u64 + 1));
            entity = entity.with_label(table_name);

            // Add each column value as a property
            for (col_name, value) in data.result_columns.iter().zip(values.into_iter()) {
                entity.set_property(col_name, value);
            }

            entity
        })
        .collect();

    Ok(Some(entities))
}

/// Build a planner catalog from the schema for index selection.
///
/// This queries the schema to find all available indexes and creates
/// a catalog that the physical planner can use to choose index scans.
fn build_planner_catalog<T: Transaction>(tx: &DatabaseTransaction<T>) -> Result<PlannerCatalog> {
    let mut catalog = PlannerCatalog::new();

    // Get all indexes from the schema
    let index_names = SchemaManager::list_indexes(tx).unwrap_or_default();

    for name in index_names {
        if let Ok(Some(schema)) = SchemaManager::get_index(tx, &name) {
            // Convert schema index type to planner index type
            let index_type = match schema.using.as_deref() {
                Some("hnsw" | "HNSW") => IndexType::Hnsw,
                Some("hash" | "HASH") => IndexType::Hash,
                _ => IndexType::BTree, // Default to B-tree
            };

            // Only add B-tree indexes for now (HNSW handled separately)
            if index_type == IndexType::BTree {
                let columns: Vec<String> = schema.columns.iter().map(|c| c.expr.clone()).collect();

                let index_info = IndexInfo::btree(&schema.name, &schema.table, columns);
                catalog = catalog.with_index(index_info);
            }
        }
    }

    Ok(catalog)
}

/// Try to execute a query using the physical plan's index scans.
///
/// Returns `Ok(Some(result))` if the physical plan contains index scans that were used,
/// or `Ok(None)` to fall back to logical plan execution.
fn try_execute_from_physical<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    physical: &PhysicalPlan,
    logical: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<Option<ResultSet>> {
    use super::index_scan::{execute_index_range_scan, execute_index_scan};

    // Check if the physical plan uses index scans
    match physical {
        PhysicalPlan::IndexScan(scan_node) => {
            // Execute the index scan
            let entities = execute_index_scan(tx, scan_node, ctx)?;
            let columns = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, columns);
            let schema = scan.schema();
            let rows = scan.collect_rows();
            Ok(Some(ResultSet::with_rows(schema, rows)))
        }

        PhysicalPlan::IndexRangeScan(scan_node) => {
            // Execute the range scan
            let entities = execute_index_range_scan(tx, scan_node, ctx)?;
            let columns = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, columns);
            let schema = scan.schema();
            let rows = scan.collect_rows();
            Ok(Some(ResultSet::with_rows(schema, rows)))
        }

        PhysicalPlan::Filter { node, input } => {
            // Check if the input is an index scan
            if let Some(result) = try_execute_from_physical(tx, input, logical, ctx)? {
                // Apply the filter to the result
                let schema = result.schema_arc();
                let predicate = &node.predicate;
                let filtered_rows: Vec<Row> = result
                    .into_rows()
                    .into_iter()
                    .filter(|row| {
                        // Use transaction-aware subquery evaluation for EXISTS/IN/scalar subqueries
                        let val = evaluate_row_expr_with_subquery_tx(tx, predicate, row, ctx);
                        matches!(val, Value::Bool(true))
                    })
                    .collect();
                return Ok(Some(ResultSet::with_rows(schema, filtered_rows)));
            }
            Ok(None)
        }

        PhysicalPlan::Project { node, input } => {
            // Check if the input is an index scan
            if let Some(result) = try_execute_from_physical(tx, input, logical, ctx)? {
                // Apply the projection
                let has_wildcard = node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

                if has_wildcard {
                    return Ok(Some(result));
                }

                let projected_columns: Vec<String> =
                    node.exprs.iter().map(|e| expr_to_column_name(e)).collect();
                let new_schema = Arc::new(Schema::new(projected_columns.clone()));
                let result_schema = result.schema_arc();

                let rows: Vec<Row> = result
                    .rows()
                    .iter()
                    .map(|row| {
                        let values: Vec<Value> = node
                            .exprs
                            .iter()
                            .map(|expr| evaluate_expr_on_row_tx(tx, expr, row, &result_schema, ctx))
                            .collect();
                        Row::new(Arc::clone(&new_schema), values)
                    })
                    .collect();

                return Ok(Some(ResultSet::with_rows(new_schema, rows)));
            }
            Ok(None)
        }

        PhysicalPlan::ProcedureCall(node) => {
            // Execute the procedure call by dispatching to the appropriate helper
            let storage = tx.storage_ref().map_err(|e| Error::Execution(e.to_string()))?;
            let result =
                execute_procedure_call(storage, &node.procedure_name, &node.arguments, ctx)?;

            // Convert RowBatch to ResultSet
            let schema = result.schema_arc();
            let rows = result.into_rows();
            Ok(Some(ResultSet::with_rows(schema, rows)))
        }

        _ => {
            // No index scan in this plan - fall back to logical execution
            Ok(None)
        }
    }
}

/// Execute a procedure call by dispatching to the appropriate helper function.
fn execute_procedure_call<T: Transaction>(
    tx: &T,
    procedure_name: &str,
    arguments: &[LogicalExpr],
    ctx: &ExecutionContext,
) -> Result<manifoldb_query::RowBatch> {
    // Helper to evaluate argument expressions to Values
    let eval_args: Vec<Value> = arguments
        .iter()
        .map(|arg| evaluate_literal_expr(arg, ctx).unwrap_or(Value::Null))
        .collect();

    // Helper functions to extract typed arguments
    let get_int = |idx: usize| -> Option<i64> {
        eval_args.get(idx).and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
    };
    let get_float = |idx: usize| -> Option<f64> {
        eval_args.get(idx).and_then(|v| match v {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        })
    };
    let get_string = |idx: usize| -> Option<&str> {
        eval_args.get(idx).and_then(|v| match v {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
    };
    let get_array = |idx: usize| -> Option<&[Value]> {
        eval_args.get(idx).and_then(|v| match v {
            Value::Array(arr) => Some(arr.as_slice()),
            _ => None,
        })
    };

    // Dispatch based on procedure name
    let result = match procedure_name {
        // Centrality algorithms
        "algo.pageRank" => {
            let damping = get_float(0).unwrap_or(0.85);
            let max_iter = get_int(1).unwrap_or(100) as usize;
            execute_pagerank_with_tx(tx, damping, max_iter)
        }
        "algo.betweennessCentrality" => {
            let normalized = eval_args.first().map_or(true, |v| matches!(v, Value::Bool(true)));
            let endpoints = eval_args.get(1).is_some_and(|v| matches!(v, Value::Bool(true)));
            execute_betweenness_with_tx(tx, normalized, endpoints)
        }
        "algo.closenessCentrality" => {
            let harmonic = eval_args.first().is_some_and(|v| matches!(v, Value::Bool(true)));
            execute_closeness_with_tx(tx, harmonic)
        }
        "algo.degreeCentrality" => {
            let direction = get_string(0);
            execute_degree_with_tx(tx, direction)
        }
        "algo.eigenvectorCentrality" => {
            let max_iter = get_int(0).unwrap_or(100) as usize;
            let tolerance = get_float(1).unwrap_or(1e-6);
            execute_eigenvector_with_tx(tx, max_iter, tolerance)
        }

        // Community detection
        "algo.louvain" => {
            let max_iter = get_int(0).unwrap_or(10) as usize;
            let tolerance = get_float(1).unwrap_or(0.0001);
            let weight_prop = get_string(2);
            execute_louvain_with_tx(tx, max_iter, tolerance, weight_prop)
        }
        "algo.labelPropagation" => {
            let max_iter = get_int(0).unwrap_or(10) as usize;
            execute_label_propagation_with_tx(tx, max_iter)
        }
        "algo.connectedComponents" => {
            let mode = get_string(0).unwrap_or("weak");
            execute_connected_components_with_tx(tx, mode)
        }
        "algo.stronglyConnectedComponents" => execute_strongly_connected_with_tx(tx),

        // Traversal algorithms
        "algo.bfs" => {
            let start_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.bfs requires startNode argument".to_string())
            })?;
            let edge_type = get_string(1);
            let direction = get_string(2);
            let max_depth = get_int(3);
            execute_bfs_with_tx(tx, start_id, edge_type, direction, max_depth)
        }
        "algo.dfs" => {
            let start_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.dfs requires startNode argument".to_string())
            })?;
            let edge_type = get_string(1);
            let direction = get_string(2);
            let max_depth = get_int(3);
            execute_dfs_with_tx(tx, start_id, edge_type, direction, max_depth)
        }

        // Path finding algorithms
        "algo.shortestPath" => {
            let source_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.shortestPath requires source argument".to_string())
            })?;
            let target_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.shortestPath requires target argument".to_string())
            })?;
            let edge_type = get_string(2);
            let max_depth = get_int(3);
            execute_shortest_path_with_tx(tx, source_id, target_id, edge_type, max_depth)
        }
        "algo.dijkstra" => {
            let source_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.dijkstra requires source argument".to_string())
            })?;
            let target_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.dijkstra requires target argument".to_string())
            })?;
            let weight_prop = get_string(2);
            let default_weight = get_float(3).unwrap_or(1.0);
            let max_weight = get_float(4);
            execute_dijkstra_with_tx(
                tx,
                source_id,
                target_id,
                weight_prop,
                default_weight,
                max_weight,
            )
        }
        "algo.astar" => {
            let source_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.astar requires source argument".to_string())
            })?;
            let target_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.astar requires target argument".to_string())
            })?;
            let weight_prop = get_string(2);
            let lat_prop = get_string(3);
            let lon_prop = get_string(4);
            let max_cost = get_float(5);
            execute_astar_with_tx(
                tx,
                source_id,
                target_id,
                weight_prop,
                lat_prop,
                lon_prop,
                max_cost,
            )
        }
        "algo.allShortestPaths" => {
            let source_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.allShortestPaths requires source argument".to_string())
            })?;
            let target_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.allShortestPaths requires target argument".to_string())
            })?;
            let edge_type = get_string(2);
            let max_depth = get_int(3);
            execute_all_shortest_paths_with_tx(tx, source_id, target_id, edge_type, max_depth)
        }
        "algo.sssp" => {
            let source_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.sssp requires source argument".to_string())
            })?;
            let weight_prop = get_string(1);
            let max_weight = get_float(2);
            execute_sssp_with_tx(tx, source_id, weight_prop, max_weight)
        }

        // Similarity algorithms
        "algo.nodeSimilarity" => {
            let label = get_string(0);
            let edge_type = get_string(1);
            let top_k = get_int(2);
            let similarity_cutoff = get_float(3).unwrap_or(0.0);
            execute_node_similarity_with_tx(tx, label, edge_type, top_k, similarity_cutoff)
        }
        "algo.jaccard" => {
            let node1_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.jaccard requires node1 argument".to_string())
            })?;
            let node2_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.jaccard requires node2 argument".to_string())
            })?;
            let edge_type = get_string(2);
            execute_jaccard_with_tx(tx, node1_id, node2_id, edge_type)
        }
        "algo.overlap" => {
            let node1_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.overlap requires node1 argument".to_string())
            })?;
            let node2_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.overlap requires node2 argument".to_string())
            })?;
            let edge_type = get_string(2);
            execute_overlap_with_tx(tx, node1_id, node2_id, edge_type)
        }
        "algo.cosine" => {
            let node1_id = get_int(0).ok_or_else(|| {
                Error::Execution("algo.cosine requires node1 argument".to_string())
            })?;
            let node2_id = get_int(1).ok_or_else(|| {
                Error::Execution("algo.cosine requires node2 argument".to_string())
            })?;
            let properties = get_array(2).unwrap_or(&[]);
            execute_cosine_with_tx(tx, node1_id, node2_id, properties)
        }

        _ => {
            return Err(Error::Execution(format!("Unknown procedure: {procedure_name}")));
        }
    };

    result.map_err(|e| Error::Execution(format!("Procedure execution failed: {e}")))
}

/// Execute a physical plan and return the result set.
fn execute_physical_plan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    physical: &PhysicalPlan,
    logical: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Try to execute index scans from the physical plan if available
    if let Some(result) = try_execute_from_physical(tx, physical, logical, ctx)? {
        return Ok(result);
    }

    // Fall back to logical plan execution
    match logical {
        LogicalPlan::Project { node, input } => {
            // Check if input is an aggregate - if so, handle specially
            if let LogicalPlan::Aggregate { node: agg_node, input: agg_input } = input.as_ref() {
                // Execute aggregate and project the results
                let result = execute_aggregate(tx, agg_node, agg_input, ctx)?;

                // If projecting only aggregate columns, return as-is
                // Otherwise, we need to project from the aggregate result
                let has_wildcard = node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));
                if has_wildcard {
                    return Ok(result);
                }

                // Project from aggregate result
                let projected_columns: Vec<String> =
                    node.exprs.iter().map(|e| expr_to_column_name(e)).collect();
                let new_schema = Arc::new(Schema::new(projected_columns.clone()));
                let result_schema = result.schema_arc();

                let rows: Vec<Row> = result
                    .rows()
                    .iter()
                    .map(|row| {
                        let values: Vec<Value> = node
                            .exprs
                            .iter()
                            .map(|expr| evaluate_expr_on_row_tx(tx, expr, row, &result_schema, ctx))
                            .collect();
                        Row::new(Arc::clone(&new_schema), values)
                    })
                    .collect();

                return Ok(ResultSet::with_rows(new_schema, rows));
            }

            // Check if input contains a JOIN - if so, execute through the join path
            if contains_join(input) {
                return execute_join_projection(tx, &node.exprs, input, ctx);
            }

            // Check if input contains a graph traversal - if so, execute through the graph path
            if contains_graph(input) {
                return execute_graph_projection(tx, &node.exprs, input, ctx);
            }

            // Handle Values nodes specially (CTEs with literal values)
            if let LogicalPlan::Values(values_node) = input.as_ref() {
                return execute_values_projection(values_node, node, ctx);
            }

            // Handle nested projections (e.g., SELECT * FROM (SELECT 1 AS id))
            // This is common for CTEs where the CTE has column aliases
            if let LogicalPlan::Project { node: inner_node, input: inner_input } = input.as_ref() {
                if let LogicalPlan::Values(values_node) = inner_input.as_ref() {
                    // First execute the inner projection on the values
                    let inner_result = execute_values_projection(values_node, inner_node, ctx)?;

                    // Then apply the outer projection (typically SELECT *)
                    let has_wildcard =
                        node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));
                    if has_wildcard {
                        return Ok(inner_result);
                    }

                    // Apply outer projection
                    let projected_columns: Vec<String> =
                        node.exprs.iter().map(|e| expr_to_column_name(e)).collect();
                    let new_schema = Arc::new(Schema::new(projected_columns.clone()));
                    let inner_schema = inner_result.schema_arc();

                    let rows: Vec<Row> = inner_result
                        .rows()
                        .iter()
                        .map(|row| {
                            let values: Vec<Value> = node
                                .exprs
                                .iter()
                                .map(|expr| {
                                    evaluate_expr_on_row_tx(tx, expr, row, &inner_schema, ctx)
                                })
                                .collect();
                            Row::new(Arc::clone(&new_schema), values)
                        })
                        .collect();

                    return Ok(ResultSet::with_rows(new_schema, rows));
                }
            }

            // First execute the input
            let input_result = execute_logical_plan(tx, input, ctx)?;

            // Check if we have a wildcard projection (SELECT *)
            let has_wildcard = node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

            if has_wildcard {
                // For SELECT *, expand to all columns from the entities
                let columns = collect_all_columns(&input_result);
                let scan = StorageScan::new(input_result, columns);
                let schema = scan.schema();
                let rows = scan.collect_rows();
                Ok(ResultSet::with_rows(schema, rows))
            } else {
                // Normal projection
                let projected_columns: Vec<String> =
                    node.exprs.iter().map(|e| expr_to_column_name(e)).collect();

                let schema = Arc::new(Schema::new(projected_columns.clone()));
                let mut rows = Vec::new();

                for entity in &input_result {
                    let values: Vec<Value> = node
                        .exprs
                        .iter()
                        .map(|expr| evaluate_expr_tx(tx, expr, entity, ctx))
                        .collect();
                    rows.push(Row::new(Arc::clone(&schema), values));
                }

                Ok(ResultSet::with_rows(schema, rows))
            }
        }

        LogicalPlan::Aggregate { node, input } => {
            // Execute aggregate directly (not wrapped in Project)
            execute_aggregate(tx, node, input, ctx)
        }

        LogicalPlan::Join { node, left, right } => execute_join(tx, node, left, right, ctx),

        LogicalPlan::Distinct { input, .. } => {
            // Execute DISTINCT by deduplicating the input result
            execute_distinct(tx, input, ctx)
        }

        LogicalPlan::Union { node, inputs } => {
            // Execute UNION / UNION ALL using UnionOp
            execute_union(tx, node, inputs, ctx)
        }

        LogicalPlan::SetOp { node, left, right } => {
            // Execute INTERSECT / EXCEPT using SetOpOp
            execute_set_op(tx, node, left, right, ctx)
        }

        LogicalPlan::Expand { node, input } => {
            // Execute graph expansion
            execute_expand(tx, node, input, ctx)
        }

        LogicalPlan::PathScan { node, input } => {
            // Execute graph path scan
            execute_path_scan(tx, node, input, ctx)
        }

        LogicalPlan::AnnSearch { node, input } => {
            // Execute ANN (approximate nearest neighbor) search using BruteForceSearchOp
            execute_ann_search(tx, node, input, ctx)
        }

        LogicalPlan::VectorDistance { node, input } => {
            // Execute vector distance computation
            execute_vector_distance(tx, node, input, ctx)
        }

        _ => {
            // For other plan types, execute and return
            let entities = execute_logical_plan(tx, logical, ctx)?;
            let columns = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, columns);
            let schema = scan.schema();
            let rows = scan.collect_rows();

            Ok(ResultSet::with_rows(schema, rows))
        }
    }
}

/// Execute an aggregate logical plan and return the result set.
fn execute_aggregate<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &manifoldb_query::plan::logical::AggregateNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute the input plan to get entities
    let entities = execute_logical_plan(tx, input, ctx)?;

    // Collect all columns from entities to build the schema for the input operator
    let columns = collect_all_columns(&entities);

    // Convert entities to rows for the operator
    let scan = StorageScan::new(entities, columns.clone());
    let rows: Vec<Vec<Value>> = scan.collect_values();

    // Create a ValuesOp as input to the aggregate operator
    let input_op: Box<dyn Operator> = Box::new(ValuesOp::with_columns(columns, rows));

    // Create the HashAggregateOp
    let mut agg_op = HashAggregateOp::new(
        node.group_by.clone(),
        node.aggregates.clone(),
        node.having.clone(),
        input_op,
    );

    // Execute the aggregate operator
    agg_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let schema = agg_op.schema();
    let mut result_rows = Vec::new();

    while let Some(row) = agg_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        result_rows.push(row);
    }

    agg_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Execute a DISTINCT query by deduplicating the input result.
///
/// Uses `HashAggregateOp` with all columns as group-by keys and no aggregates.
/// This is a standard technique for implementing DISTINCT.
fn execute_distinct<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Check if input contains a JOIN - handle through join path first
    if contains_join(input) {
        let join_result = execute_join_plan(tx, input, ctx)?;
        return deduplicate_result_set(join_result, ctx);
    }

    // For projections, execute the projection first, then deduplicate
    if let LogicalPlan::Project { node, input: proj_input } = input {
        // Execute the input entities first
        let entities = execute_logical_plan(tx, proj_input, ctx)?;

        // Check for wildcard projection
        let has_wildcard = node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

        // Build the projected rows
        let (columns, rows): (Vec<String>, Vec<Vec<Value>>) = if has_wildcard {
            // For SELECT *, use all entity columns
            let cols = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, cols.clone());
            (cols, scan.collect_values())
        } else {
            // For specific columns, project
            let cols: Vec<String> = node.exprs.iter().map(|e| expr_to_column_name(e)).collect();
            let row_vals: Vec<Vec<Value>> = entities
                .iter()
                .map(|entity| {
                    node.exprs.iter().map(|expr| evaluate_expr(expr, entity, ctx)).collect()
                })
                .collect();
            (cols, row_vals)
        };

        // Deduplicate using HashAggregateOp
        let group_by: Vec<LogicalExpr> = columns.iter().map(|c| LogicalExpr::column(c)).collect();
        let input_op: Box<dyn Operator> = Box::new(ValuesOp::with_columns(columns, rows));
        let mut agg_op = HashAggregateOp::new(group_by, vec![], None, input_op);

        agg_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;
        let schema = agg_op.schema();
        let mut result_rows = Vec::new();
        while let Some(row) = agg_op.next().map_err(|e| Error::Execution(e.to_string()))? {
            result_rows.push(row);
        }
        agg_op.close().map_err(|e| Error::Execution(e.to_string()))?;

        return Ok(ResultSet::with_rows(schema, result_rows));
    }

    // For other inputs, execute and deduplicate
    let entities = execute_logical_plan(tx, input, ctx)?;

    // Collect all columns from entities to build the schema
    let columns = collect_all_columns(&entities);

    // Convert entities to rows for the operator
    let scan = StorageScan::new(entities, columns.clone());
    let rows: Vec<Vec<Value>> = scan.collect_values();

    // Create column expressions for group-by (all columns become group-by keys)
    let group_by: Vec<LogicalExpr> = columns.iter().map(|c| LogicalExpr::column(c)).collect();

    // Create a ValuesOp as input
    let input_op: Box<dyn Operator> = Box::new(ValuesOp::with_columns(columns, rows));

    // Create HashAggregateOp with all columns as group-by and no aggregates
    let mut agg_op = HashAggregateOp::new(
        group_by,
        vec![], // No aggregate functions - just deduplication
        None,   // No HAVING clause
        input_op,
    );

    // Execute the operator
    agg_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let schema = agg_op.schema();
    let mut result_rows = Vec::new();

    while let Some(row) = agg_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        result_rows.push(row);
    }

    agg_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Deduplicate a result set using HashAggregateOp.
fn deduplicate_result_set(result: ResultSet, ctx: &ExecutionContext) -> Result<ResultSet> {
    let schema = result.schema_arc();
    let columns: Vec<String> = schema.columns().iter().map(|c| (*c).to_string()).collect();

    // Create column expressions for group-by (all columns become group-by keys)
    let group_by: Vec<LogicalExpr> = columns.iter().map(|c| LogicalExpr::column(c)).collect();

    // Convert rows to values
    let rows: Vec<Vec<Value>> = result
        .into_rows()
        .into_iter()
        .map(|row| {
            (0..row.schema().columns().len())
                .map(|i| row.get(i).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();

    // Create a ValuesOp as input
    let input_op: Box<dyn Operator> = Box::new(ValuesOp::with_columns(columns, rows));

    // Create HashAggregateOp with all columns as group-by and no aggregates
    let mut agg_op = HashAggregateOp::new(
        group_by,
        vec![], // No aggregate functions
        None,   // No HAVING clause
        input_op,
    );

    // Execute the operator
    agg_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let output_schema = agg_op.schema();
    let mut result_rows = Vec::new();

    while let Some(row) = agg_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        result_rows.push(row);
    }

    agg_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(output_schema, result_rows))
}

/// Execute a UNION query using UnionOp.
///
/// Handles both UNION (with deduplication) and UNION ALL.
fn execute_union<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &UnionNode,
    inputs: &[LogicalPlan],
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute each input plan and convert to operators
    let input_ops: Vec<Box<dyn Operator>> =
        inputs.iter().map(|input| plan_to_operator(tx, input, ctx)).collect::<Result<Vec<_>>>()?;

    if input_ops.is_empty() {
        return Ok(ResultSet::new(Arc::new(Schema::empty())));
    }

    // Create UnionOp
    let mut union_op = UnionOp::new(input_ops, node.all);

    // Execute the operator
    union_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let schema = union_op.schema();
    let mut result_rows = Vec::new();

    while let Some(row) = union_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        result_rows.push(row);
    }

    union_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Execute a set operation (INTERSECT/EXCEPT) using SetOpOp.
fn execute_set_op<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &SetOpNode,
    left: &LogicalPlan,
    right: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute both input plans and convert to operators
    let left_op = plan_to_operator(tx, left, ctx)?;
    let right_op = plan_to_operator(tx, right, ctx)?;

    // Create SetOpOp
    let mut set_op = SetOpOp::new(node.op_type, left_op, right_op);

    // Execute the operator
    set_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let schema = set_op.schema();
    let mut result_rows = Vec::new();

    while let Some(row) = set_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        result_rows.push(row);
    }

    set_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Execute a graph expand query by directly calling graph traversal APIs.
///
/// This executes single-hop or variable-length graph traversals.
fn execute_expand<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &ExpandNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute the input plan to get starting nodes
    let input_result = execute_graph_plan(tx, input, ctx)?;

    // Extract source node IDs from the input
    let source_nodes = graph_accessor::extract_source_nodes(input_result, &node.src_var);

    // Execute the expand operation
    graph_accessor::execute_expand_operation(tx, node, source_nodes)
}

/// Execute a graph path scan query by executing each step sequentially.
///
/// This executes multi-step path pattern matching by chaining expand operations.
fn execute_path_scan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &PathScanNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute the input plan to get starting nodes
    let mut current_result = execute_graph_plan(tx, input, ctx)?;

    // Execute each path step in sequence
    for step in &node.steps {
        // Extract source nodes from the current result
        let source_nodes =
            graph_accessor::extract_source_nodes(current_result, &step.expand.src_var);

        // Execute the expand for this step
        current_result = graph_accessor::execute_expand_operation(tx, &step.expand, source_nodes)?;
    }

    Ok(current_result)
}

/// Execute an ANN (approximate nearest neighbor) search.
///
/// Uses HNSW index when available (with in-traversal filtering for efficiency),
/// otherwise falls back to `BruteForceSearchOp` for exact k-NN search.
fn execute_ann_search<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &AnnSearchNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Try to find an HNSW index for this column
    let table_name = extract_table_name(input);
    let index_name = if let Some(table) = table_name {
        crate::vector::find_index_for_column(tx, &table, &node.vector_column).ok().flatten()
    } else {
        None
    };

    // If we have an index, use it for efficient filtered search
    if let Some(idx_name) = index_name {
        return execute_ann_search_with_index(tx, node, input, ctx, &idx_name);
    }

    // Fall back to brute force search
    execute_ann_search_brute_force(tx, node, input, ctx)
}

/// Execute ANN search using an HNSW index.
fn execute_ann_search_with_index<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &AnnSearchNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
    index_name: &str,
) -> Result<ResultSet> {
    use manifoldb_vector::types::Embedding;
    use std::collections::HashSet;

    // Evaluate the query vector
    let query_value = evaluate_literal_expr(&node.query_vector, ctx)
        .map_err(|_| Error::Execution("Could not evaluate query vector".to_string()))?;

    let query_vec = match query_value {
        Value::Vector(v) => v,
        _ => {
            return Err(Error::Execution("Query must be a vector".to_string()));
        }
    };

    let query = Embedding::new(query_vec).map_err(|e| Error::Execution(e.to_string()))?;

    // Execute the input plan to get entities that pass any non-vector filters
    let entities = execute_logical_plan(tx, input, ctx)?;

    // Collect all columns for the output schema
    let mut columns = collect_all_columns(&entities);

    // Create a set of entity IDs that pass the input filters (pre-filter stage)
    let matching_ids: HashSet<manifoldb_core::EntityId> = entities.iter().map(|e| e.id).collect();

    // Create entity lookup map for building result rows
    let entity_map: std::collections::HashMap<manifoldb_core::EntityId, &manifoldb_core::Entity> =
        entities.iter().map(|e| (e.id, e)).collect();

    // If there's a filter in the AnnSearchNode, we need to combine it with entity matching
    let has_filter = node.filter.is_some() || !matching_ids.is_empty();

    let search_result = if has_filter && !entity_map.is_empty() {
        // Create a predicate that checks if an entity ID is in our matching set
        // AND evaluates the ANN search filter if present
        let predicate = |id: manifoldb_core::EntityId| {
            // Must be in the set of entities from input
            if !matching_ids.contains(&id) {
                return false;
            }
            // If there's an additional filter, evaluate it
            if let Some(ref filter_expr) = node.filter {
                if let Some(entity) = entity_map.get(&id) {
                    return evaluate_predicate(filter_expr, entity, ctx);
                }
                return false;
            }
            true
        };

        // Use filtered HNSW search
        crate::vector::search_index_filtered(
            tx,
            index_name,
            &query,
            node.k,
            predicate,
            node.params.ef_search,
            None,
        )
        .map_err(|e| Error::Execution(e.to_string()))?
    } else {
        // Use regular HNSW search (no filter)
        crate::vector::search_index(tx, index_name, &query, node.k, node.params.ef_search)
            .map_err(|e| Error::Execution(e.to_string()))?
    };

    // Add distance column if requested
    if node.include_distance {
        let distance_col = node.distance_alias.clone().unwrap_or_else(|| "distance".to_string());
        columns.push(distance_col);
    }

    let schema = Arc::new(Schema::new(columns.clone()));
    let mut result_rows = Vec::new();

    // Get non-distance columns for building rows
    let data_columns: Vec<&String> = if node.include_distance {
        columns.iter().take(columns.len() - 1).collect()
    } else {
        columns.iter().collect()
    };

    for result in search_result {
        if let Some(entity) = entity_map.get(&result.entity_id) {
            let mut values: Vec<Value> = data_columns
                .iter()
                .map(|col| {
                    if *col == "_rowid" {
                        Value::Int(entity.id.as_u64() as i64)
                    } else {
                        entity.get_property(col).cloned().unwrap_or(Value::Null)
                    }
                })
                .collect();

            if node.include_distance {
                values.push(Value::Float(f64::from(result.distance)));
            }

            result_rows.push(Row::new(Arc::clone(&schema), values));
        }
    }

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Execute ANN search using brute force (no index).
fn execute_ann_search_brute_force<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &AnnSearchNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute the input plan to get entities
    let entities = execute_logical_plan(tx, input, ctx)?;

    // Collect all columns from entities to build the schema for the input operator
    let columns = collect_all_columns(&entities);

    // Convert entities to rows for the operator
    let scan = StorageScan::new(entities, columns.clone());
    let rows: Vec<Vec<Value>> = scan.collect_values();

    // Create a ValuesOp as input to the vector search operator
    let input_op: Box<dyn Operator> = Box::new(ValuesOp::with_columns(columns, rows));

    // Create the BruteForceSearchOp for k-NN search
    let mut search_op = BruteForceSearchOp::new(
        node.vector_column.clone(),
        node.query_vector.clone(),
        node.metric,
        node.k,
        node.include_distance,
        node.distance_alias.clone(),
        input_op,
    );

    // Execute the search operator
    search_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let schema = search_op.schema();
    let mut result_rows = Vec::new();

    while let Some(row) = search_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        result_rows.push(row);
    }

    search_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Extract the table name from a logical plan's scan node.
fn extract_table_name(plan: &LogicalPlan) -> Option<String> {
    match plan {
        LogicalPlan::Scan(node) => Some(node.table_name.clone()),
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Alias { input, .. } => extract_table_name(input),
        _ => None,
    }
}

/// Execute a vector distance computation.
///
/// Computes the distance between vectors and adds a distance column to the result.
fn execute_vector_distance<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &VectorDistanceNode,
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute the input plan to get entities
    let entities = execute_logical_plan(tx, input, ctx)?;

    // Collect all columns from entities
    let mut columns = collect_all_columns(&entities);

    // Add distance column to the schema
    let distance_col = node.alias.clone().unwrap_or_else(|| "distance".to_string());
    columns.push(distance_col.clone());

    let schema = Arc::new(Schema::new(columns.clone()));
    let mut result_rows = Vec::new();

    for entity in &entities {
        // Evaluate both vector expressions for this entity
        let left_val = evaluate_expr(&node.left, entity, ctx);
        let right_val = evaluate_expr(&node.right, entity, ctx);

        // Compute the distance
        let distance = compute_vector_distance(&left_val, &right_val, &node.metric);

        // Build the row with entity properties + distance
        let mut values: Vec<Value> = columns[..columns.len() - 1]
            .iter()
            .map(|col| {
                if col == "_rowid" {
                    Value::Int(entity.id.as_u64() as i64)
                } else {
                    entity.get_property(col).cloned().unwrap_or(Value::Null)
                }
            })
            .collect();
        values.push(distance);

        result_rows.push(Row::new(Arc::clone(&schema), values));
    }

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Compute vector distance using the specified metric.
fn compute_vector_distance(left: &Value, right: &Value, metric: &DistanceMetric) -> Value {
    match (left, right) {
        (Value::Vector(a), Value::Vector(b)) if a.len() == b.len() => {
            let dist = match metric {
                DistanceMetric::Euclidean => {
                    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
                }
                DistanceMetric::Cosine => {
                    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm_a == 0.0 || norm_b == 0.0 {
                        f32::MAX
                    } else {
                        1.0 - (dot / (norm_a * norm_b))
                    }
                }
                DistanceMetric::InnerProduct => {
                    // Negative inner product for distance ordering
                    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
                }
                DistanceMetric::Manhattan => {
                    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
                }
                DistanceMetric::Hamming => {
                    a.iter().zip(b.iter()).filter(|(x, y)| (*x - *y).abs() > f32::EPSILON).count()
                        as f32
                }
            };
            Value::Float(f64::from(dist))
        }
        _ => Value::Null,
    }
}

/// Execute a projection over a Values node (for CTEs with literal values).
///
/// This handles cases like `WITH cte AS (SELECT 1 AS id, 'name' AS name) SELECT * FROM cte`
/// where the CTE input is a Values node containing literal expressions.
fn execute_values_projection(
    values_node: &ValuesNode,
    project_node: &ProjectNode,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Get column names from the values node
    let column_names = values_node.column_names.clone().unwrap_or_else(|| {
        (0..values_node.rows.first().map(|r| r.len()).unwrap_or(0))
            .map(|i| format!("col{}", i))
            .collect()
    });

    // Build schema from the column names
    let schema = Arc::new(Schema::new(column_names.clone()));

    // Evaluate each row's expressions to produce actual values
    let mut rows = Vec::new();
    for row_exprs in &values_node.rows {
        let values: Vec<Value> =
            row_exprs.iter().map(|expr| evaluate_values_literal_expr(expr)).collect();
        rows.push(Row::new(Arc::clone(&schema), values));
    }

    // If projection is a wildcard, return all columns
    let has_wildcard = project_node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));
    if has_wildcard {
        return Ok(ResultSet::with_rows(schema, rows));
    }

    // Apply projection
    let projected_columns: Vec<String> =
        project_node.exprs.iter().map(|e| expr_to_column_name(e)).collect();
    let new_schema = Arc::new(Schema::new(projected_columns.clone()));

    let projected_rows: Vec<Row> = rows
        .iter()
        .map(|row| {
            let values: Vec<Value> = project_node
                .exprs
                .iter()
                .map(|expr| evaluate_expr_on_row(expr, row, &schema, ctx))
                .collect();
            Row::new(Arc::clone(&new_schema), values)
        })
        .collect();

    Ok(ResultSet::with_rows(new_schema, projected_rows))
}

/// Evaluate a literal expression to a Value (for Values node projection).
fn evaluate_values_literal_expr(expr: &LogicalExpr) -> Value {
    match expr {
        LogicalExpr::Literal(lit) => match lit {
            Literal::Integer(n) => Value::Int(*n),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
            Literal::Boolean(b) => Value::Bool(*b),
            Literal::Null => Value::Null,
            Literal::Vector(v) => Value::Vector(v.clone()),
            Literal::MultiVector(v) => Value::MultiVector(v.clone()),
        },
        LogicalExpr::Alias { expr, .. } => evaluate_values_literal_expr(expr),
        _ => Value::Null,
    }
}

/// Execute a projection over a graph traversal result.
fn execute_graph_projection<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    exprs: &[LogicalExpr],
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // First execute the graph traversal
    let graph_result = execute_graph_plan(tx, input, ctx)?;

    // Check for wildcard projection
    let has_wildcard = exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

    if has_wildcard {
        // Return all columns from the graph result
        Ok(graph_result)
    } else {
        // Project specific columns
        let projected_columns: Vec<String> = exprs.iter().map(|e| expr_to_column_name(e)).collect();
        let new_schema = Arc::new(Schema::new(projected_columns.clone()));

        let mut projected_rows = Vec::new();
        for row in graph_result.rows() {
            let values: Vec<Value> =
                exprs.iter().map(|expr| evaluate_row_expr(expr, row)).collect();
            projected_rows.push(Row::new(Arc::clone(&new_schema), values));
        }

        Ok(ResultSet::with_rows(new_schema, projected_rows))
    }
}

/// Execute a plan that may contain graph traversals, returning a ResultSet.
fn execute_graph_plan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    match plan {
        LogicalPlan::Expand { node, input } => execute_expand(tx, node, input, ctx),
        LogicalPlan::PathScan { node, input } => execute_path_scan(tx, node, input, ctx),
        LogicalPlan::Filter { node, input } => {
            // Apply filter to graph result
            let graph_result = execute_graph_plan(tx, input, ctx)?;
            let schema = graph_result.schema_arc();
            let filtered_rows: Vec<Row> = graph_result
                .into_rows()
                .into_iter()
                .filter(|row| {
                    let result = evaluate_row_expr(&node.predicate, row);
                    matches!(result, Value::Bool(true))
                })
                .collect();
            Ok(ResultSet::with_rows(schema, filtered_rows))
        }
        LogicalPlan::Sort { node, input } => {
            // Apply sort to graph result
            let graph_result = execute_graph_plan(tx, input, ctx)?;
            let schema = graph_result.schema_arc();
            let mut rows = graph_result.into_rows();

            if !node.order_by.is_empty() {
                rows.sort_by(|a, b| {
                    // Compare by each sort key in order
                    for order in &node.order_by {
                        let va = evaluate_row_expr(&order.expr, a);
                        let vb = evaluate_row_expr(&order.expr, b);
                        let cmp = compare_values(&va, &vb);
                        let cmp = if order.ascending { cmp } else { cmp.reverse() };
                        if cmp != std::cmp::Ordering::Equal {
                            return cmp;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }

            Ok(ResultSet::with_rows(schema, rows))
        }
        LogicalPlan::Limit { node, input } => {
            // Apply limit to graph result
            let graph_result = execute_graph_plan(tx, input, ctx)?;
            let schema = graph_result.schema_arc();
            let rows = graph_result.into_rows();

            let start = node.offset.unwrap_or(0);
            let end = node.limit.map(|l| start + l).unwrap_or(rows.len());
            let limited_rows: Vec<Row> = rows.into_iter().skip(start).take(end - start).collect();

            Ok(ResultSet::with_rows(schema, limited_rows))
        }
        LogicalPlan::Alias { input, .. } => {
            // Alias doesn't change the rows
            execute_graph_plan(tx, input, ctx)
        }
        LogicalPlan::Scan(scan_node) => {
            let label = &scan_node.table_name;
            let alias = scan_node.alias.as_deref();

            // Check if this is a materialized view scan
            if let Some(result) = try_execute_materialized_view_scan(tx, label, alias)? {
                return Ok(result);
            }

            // Execute a table scan and convert to result set for graph traversal input
            let alias = alias.unwrap_or(label);
            let entities = tx.iter_entities(Some(label)).map_err(Error::Transaction)?;

            // Build schema with prefixed column names
            let columns = collect_all_columns(&entities);
            let prefixed_columns: Vec<String> =
                columns.iter().map(|c| format!("{}.{}", alias, c)).collect();
            let schema = Arc::new(Schema::new(prefixed_columns.clone()));

            // Convert entities to rows
            let rows: Vec<Row> = entities
                .iter()
                .map(|entity| {
                    let values: Vec<Value> = columns
                        .iter()
                        .map(|col| {
                            if col == "_rowid" {
                                Value::Int(entity.id.as_u64() as i64)
                            } else {
                                entity.get_property(col).cloned().unwrap_or(Value::Null)
                            }
                        })
                        .collect();
                    Row::new(Arc::clone(&schema), values)
                })
                .collect();

            Ok(ResultSet::with_rows(schema, rows))
        }
        _ => {
            // Fall back to entity execution for non-graph plans
            let entities = execute_logical_plan(tx, plan, ctx)?;
            let columns = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, columns);
            let schema = scan.schema();
            let rows = scan.collect_rows();
            Ok(ResultSet::with_rows(schema, rows))
        }
    }
}

/// Convert a logical plan to a boxed operator for use in set operations.
///
/// This recursively executes the plan and wraps the results in a ValuesOp.
fn plan_to_operator<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<Box<dyn Operator>> {
    // Check if this is a JOIN - needs special handling
    if contains_join(plan) {
        let join_result = execute_join_plan(tx, plan, ctx)?;
        let schema = join_result.schema_arc();
        let columns: Vec<String> = schema.columns().iter().map(|c| (*c).to_string()).collect();
        let rows: Vec<Vec<Value>> = join_result
            .into_rows()
            .into_iter()
            .map(|row| {
                (0..row.schema().columns().len())
                    .map(|i| row.get(i).cloned().unwrap_or(Value::Null))
                    .collect()
            })
            .collect();
        return Ok(Box::new(ValuesOp::with_columns(columns, rows)));
    }

    // For Project nodes, execute the projection
    if let LogicalPlan::Project { node, input } = plan {
        // Check for wildcard
        let has_wildcard = node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

        let entities = execute_logical_plan(tx, input, ctx)?;

        let (columns, rows): (Vec<String>, Vec<Vec<Value>>) = if has_wildcard {
            let cols = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, cols.clone());
            (cols, scan.collect_values())
        } else {
            let cols: Vec<String> = node.exprs.iter().map(|e| expr_to_column_name(e)).collect();
            let row_vals: Vec<Vec<Value>> = entities
                .iter()
                .map(|entity| {
                    node.exprs.iter().map(|expr| evaluate_expr(expr, entity, ctx)).collect()
                })
                .collect();
            (cols, row_vals)
        };

        return Ok(Box::new(ValuesOp::with_columns(columns, rows)));
    }

    // For other plans, execute and convert to ValuesOp
    let entities = execute_logical_plan(tx, plan, ctx)?;
    let columns = collect_all_columns(&entities);
    let scan = StorageScan::new(entities, columns.clone());
    let rows: Vec<Vec<Value>> = scan.collect_values();

    Ok(Box::new(ValuesOp::with_columns(columns, rows)))
}

/// Evaluate an expression on a Row (for aggregate result projection).
fn evaluate_expr_on_row(
    expr: &LogicalExpr,
    row: &Row,
    schema: &Arc<Schema>,
    ctx: &ExecutionContext,
) -> Value {
    match expr {
        LogicalExpr::Literal(lit) => literal_to_value(lit),

        LogicalExpr::Column { name, .. } => {
            // Find column index in schema
            if let Some(idx) = schema.index_of(name) {
                row.get(idx).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        LogicalExpr::Parameter(idx) => {
            ctx.get_parameter(*idx as u32).cloned().unwrap_or(Value::Null)
        }

        LogicalExpr::Alias { expr, .. } => evaluate_expr_on_row(expr, row, schema, ctx),

        LogicalExpr::AggregateFunction { func, .. } => {
            // Look up the aggregate result by its name (e.g., "COUNT", "SUM")
            let name = format!("{func}");
            if let Some(idx) = schema.index_of(&name) {
                row.get(idx).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        LogicalExpr::Wildcard => Value::Null,

        _ => Value::Null,
    }
}

/// Evaluate a logical expression on a row with transaction access for subqueries.
///
/// This version handles scalar subqueries and other expressions that require
/// database access to evaluate.
fn evaluate_expr_on_row_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    expr: &LogicalExpr,
    row: &Row,
    schema: &Arc<Schema>,
    ctx: &ExecutionContext,
) -> Value {
    match expr {
        // Scalar subquery - execute and return the result
        LogicalExpr::Subquery(subquery) => evaluate_scalar_subquery_with_tx(tx, subquery, ctx),

        // Binary operations - recursively evaluate operands
        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr_on_row_tx(tx, left, row, schema, ctx);
            let rval = evaluate_expr_on_row_tx(tx, right, row, schema, ctx);
            evaluate_binary_op(op, &lval, &rval)
        }

        // Alias - unwrap and evaluate inner
        LogicalExpr::Alias { expr: inner, .. } => {
            evaluate_expr_on_row_tx(tx, inner, row, schema, ctx)
        }

        // For non-subquery expressions, delegate to the basic version
        _ => evaluate_expr_on_row(expr, row, schema, ctx),
    }
}

/// Collect all unique column names from a set of entities.
fn collect_all_columns(entities: &[Entity]) -> Vec<String> {
    if entities.is_empty() {
        return vec![];
    }

    // Start with _rowid, then add all unique property keys
    let mut cols: Vec<String> = vec!["_rowid".to_string()];
    for entity in entities {
        for key in entity.properties.keys() {
            if !cols.contains(key) {
                cols.push(key.clone());
            }
        }
    }
    cols
}

/// Execute a logical plan and return matching entities.
fn execute_logical_plan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<Vec<Entity>> {
    match plan {
        LogicalPlan::Scan(scan_node) => {
            // The table name is the entity label
            let label = &scan_node.table_name;

            // Check if this is a materialized view
            if let Some(entities) = try_execute_materialized_view_as_entities(tx, label)? {
                return Ok(entities);
            }

            let entities = tx.iter_entities(Some(label)).map_err(Error::Transaction)?;
            Ok(entities)
        }

        LogicalPlan::Filter { node, input } => {
            let entities = execute_logical_plan(tx, input, ctx)?;

            // Filter entities based on predicate (use _with_tx for subquery support)
            let predicate = &node.predicate;
            let filtered: Vec<Entity> = entities
                .into_iter()
                .filter(|entity| evaluate_predicate_with_tx(tx, predicate, entity, ctx))
                .collect();

            Ok(filtered)
        }

        LogicalPlan::Project { input, .. } => {
            // Projection doesn't change the entities, just what we extract
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Limit { node, input } => {
            let entities = execute_logical_plan(tx, input, ctx)?;

            let start = node.offset.unwrap_or(0);
            let end = node.limit.map(|l| start + l).unwrap_or(entities.len());

            Ok(entities.into_iter().skip(start).take(end - start).collect())
        }

        LogicalPlan::Sort { node, input } => {
            let mut entities = execute_logical_plan(tx, input, ctx)?;

            // Sort by all order-by expressions
            if !node.order_by.is_empty() {
                entities.sort_by(|a, b| {
                    // Compare by each sort key in order
                    for order in &node.order_by {
                        let va = evaluate_expr(&order.expr, a, ctx);
                        let vb = evaluate_expr(&order.expr, b, ctx);
                        let cmp = compare_values(&va, &vb);
                        let cmp = if order.ascending { cmp } else { cmp.reverse() };
                        if cmp != std::cmp::Ordering::Equal {
                            return cmp;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }

            Ok(entities)
        }

        LogicalPlan::Values(_) => {
            // VALUES clause doesn't read from storage
            // Just return empty for now (used for INSERT)
            Ok(Vec::new())
        }

        LogicalPlan::Empty { .. } => Ok(Vec::new()),

        LogicalPlan::Aggregate { input, .. } => {
            // Execute the input - aggregation is handled at the physical plan level
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Distinct { input, .. } => {
            // Execute the input - deduplication is handled at the physical plan level
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Alias { input, .. } => {
            // Alias is logical-only, just execute the input
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Unwind { .. } => {
            // UNWIND produces multiple rows per entity, which doesn't fit the entity model.
            // This should be executed through execute_physical_plan.
            Err(Error::Execution(
                "UNWIND queries should be executed through execute_physical_plan, not execute_logical_plan"
                    .to_string(),
            ))
        }

        LogicalPlan::Join { .. } => {
            // JOIN produces rows, not entities. Handle through execute_physical_plan.
            Err(Error::Execution(
                "JOIN queries should be executed through execute_physical_plan, not execute_logical_plan"
                    .to_string(),
            ))
        }

        LogicalPlan::SetOp { .. } => {
            // Set operations produce rows, not entities. Handle through execute_physical_plan.
            Err(Error::Execution(
                "Set operations should be executed through execute_physical_plan, not execute_logical_plan"
                    .to_string(),
            ))
        }

        LogicalPlan::Union { .. } => {
            // UNION produces rows, not entities. Handle through execute_physical_plan.
            Err(Error::Execution(
                "UNION queries should be executed through execute_physical_plan, not execute_logical_plan"
                    .to_string(),
            ))
        }

        LogicalPlan::Expand { .. } => Err(Error::Execution(
            "Graph EXPAND queries not yet supported in entity execution".to_string(),
        )),

        LogicalPlan::PathScan { .. } => Err(Error::Execution(
            "Graph path scan queries not yet supported in entity execution".to_string(),
        )),

        LogicalPlan::ShortestPath { .. } => Err(Error::Execution(
            "Shortest path queries should be executed through execute_physical_plan, not execute_logical_plan"
                .to_string(),
        )),

        LogicalPlan::AnnSearch { input, .. } => {
            // ANN search in entity context: just execute the input
            // The actual k-NN sorting is handled at the physical plan level
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::VectorDistance { input, .. } => {
            // Vector distance in entity context: just execute the input
            // Distance computation happens in expression evaluation during projection/sorting
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Insert { .. } | LogicalPlan::Update { .. } | LogicalPlan::Delete { .. } => {
            Err(Error::Execution(
                "DML statements should be executed via execute_statement, not execute_logical_plan"
                    .to_string(),
            ))
        }

        LogicalPlan::CreateTable(_)
        | LogicalPlan::AlterTable(_)
        | LogicalPlan::DropTable(_)
        | LogicalPlan::TruncateTable(_)
        | LogicalPlan::CreateIndex(_)
        | LogicalPlan::AlterIndex(_)
        | LogicalPlan::DropIndex(_)
        | LogicalPlan::CreateCollection(_)
        | LogicalPlan::DropCollection(_)
        | LogicalPlan::CreateView(_)
        | LogicalPlan::DropView(_)
        | LogicalPlan::CreateMaterializedView(_)
        | LogicalPlan::DropMaterializedView(_)
        | LogicalPlan::RefreshMaterializedView(_)
        | LogicalPlan::CreateSchema(_)
        | LogicalPlan::AlterSchema(_)
        | LogicalPlan::DropSchema(_)
        | LogicalPlan::CreateFunction(_)
        | LogicalPlan::DropFunction(_)
        | LogicalPlan::CreateTrigger(_)
        | LogicalPlan::DropTrigger(_) => Err(Error::Execution(
            "DDL statements should be executed via execute_statement, not execute_logical_plan"
                .to_string(),
        )),

        LogicalPlan::HybridSearch { input, .. } => {
            // Hybrid search in entity context: execute the input, scoring handled by physical plan
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Window { .. } => Err(Error::Execution(
            "Window functions should be executed through execute_physical_plan, not execute_logical_plan"
                .to_string(),
        )),

        LogicalPlan::RecursiveCTE { .. } => Err(Error::Execution(
            "Recursive CTEs should be executed through execute_physical_plan, not execute_logical_plan"
                .to_string(),
        )),

        LogicalPlan::GraphCreate { .. }
        | LogicalPlan::GraphMerge { .. }
        | LogicalPlan::GraphSet { .. }
        | LogicalPlan::GraphDelete { .. }
        | LogicalPlan::GraphRemove { .. }
        | LogicalPlan::GraphForeach { .. } => Err(Error::Execution(
            "Graph DML operations should be executed via execute_statement, not execute_logical_plan"
                .to_string(),
        )),

        LogicalPlan::ProcedureCall(_) => Err(Error::Execution(
            "Procedure calls should be executed through execute_physical_plan, not execute_logical_plan"
                .to_string(),
        )),

        // Transaction control statements don't return entities
        LogicalPlan::BeginTransaction(_)
        | LogicalPlan::Commit(_)
        | LogicalPlan::Rollback(_)
        | LogicalPlan::Savepoint(_)
        | LogicalPlan::ReleaseSavepoint(_)
        | LogicalPlan::SetTransaction(_) => {
            // Transaction control is handled at the session level, not here
            Ok(Vec::new())
        }

        // CALL { } inline subquery - executes subquery for each outer row
        LogicalPlan::CallSubquery { input, .. } => {
            // For this execution path, we just pass through to the input
            // The full CALL subquery semantics are handled by the physical operator
            execute_logical_plan(tx, input, ctx)
        }

        // Utility statements don't return entities
        LogicalPlan::ExplainAnalyze(_)
        | LogicalPlan::Vacuum(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Copy(_)
        | LogicalPlan::SetSession(_)
        | LogicalPlan::Show(_)
        | LogicalPlan::Reset(_)
        | LogicalPlan::ShowProcedures(_) => {
            // Utility statements are handled at the session level, not here
            Ok(Vec::new())
        }

        // MERGE SQL is handled separately by execute_merge_sql
        LogicalPlan::MergeSql { .. } => Ok(Vec::new()),
    }
}

/// Execute an INSERT statement.
fn execute_insert<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    columns: &[String],
    input: &LogicalPlan,
    on_conflict: Option<&manifoldb_query::plan::logical::LogicalOnConflict>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};
    use manifoldb_query::plan::logical::LogicalConflictAction;
    use manifoldb_vector::types::VectorData;

    let mut count = 0;

    // Check if this is a collection with named vectors
    let collection = CollectionName::new(table)
        .ok()
        .and_then(|name| CollectionManager::get(tx, &name).ok().flatten());

    // If columns not specified, look up from table schema
    let resolved_columns: Vec<String> = if columns.is_empty() {
        // Try to get table schema for column names
        match SchemaManager::get_table(tx, table) {
            Ok(Some(schema)) => schema.columns.iter().map(|c| c.name.clone()).collect(),
            Ok(None) => {
                // No schema exists - for backward compatibility with entity-only inserts,
                // allow insertion but no properties will be set
                Vec::new()
            }
            Err(e) => {
                return Err(Error::Execution(format!("Failed to get table schema: {e}")));
            }
        }
    } else {
        columns.to_vec()
    };

    // Extract values from the input plan
    if let LogicalPlan::Values(values_node) = input {
        for row_exprs in &values_node.rows {
            // First evaluate all values for this row
            let mut row_values: HashMap<String, Value> = HashMap::new();
            let mut vectors_to_store: Vec<(String, VectorData)> = Vec::new();

            for (i, col) in resolved_columns.iter().enumerate() {
                if let Some(expr) = row_exprs.get(i) {
                    let value = evaluate_literal_expr(expr, ctx)?;

                    // For collections, check if this column is a named vector
                    if let Some(ref coll) = collection {
                        if coll.has_vector(col) {
                            // Convert Value to VectorData and store separately
                            if let Some(vector_data) = value_to_vector_data(&value) {
                                vectors_to_store.push((col.clone(), vector_data));
                            }
                            // Don't store vector in entity properties for collections
                            continue;
                        }
                    }

                    row_values.insert(col.clone(), value);
                }
            }

            // Check for conflicts if ON CONFLICT is specified
            let (existing_entity, conflict_found) = if let Some(oc) = on_conflict {
                find_conflicting_entity(tx, table, &oc.target, &row_values)?
            } else {
                (None, false)
            };

            // Handle conflict based on the action
            if conflict_found {
                match on_conflict {
                    Some(oc) => match &oc.action {
                        LogicalConflictAction::DoNothing => {
                            // Skip this row - don't increment count
                            continue;
                        }
                        LogicalConflictAction::DoUpdate { assignments, where_clause } => {
                            // Update the existing entity
                            if let Some(existing) = existing_entity {
                                // Check WHERE clause if present
                                let should_update = if let Some(where_expr) = where_clause {
                                    evaluate_predicate(where_expr, &existing, ctx)
                                } else {
                                    true
                                };

                                if should_update {
                                    // Clone before modification for index maintenance
                                    let old_entity = existing.clone();
                                    let mut new_entity = existing;

                                    // Apply assignments
                                    for (col, expr) in assignments {
                                        let value = evaluate_expr(expr, &new_entity, ctx);
                                        new_entity.properties.insert(col.clone(), value);
                                    }

                                    // Store updated entity
                                    tx.put_entity(&new_entity).map_err(Error::Transaction)?;

                                    // Update property indexes
                                    super::index_maintenance::EntityIndexMaintenance::on_update(
                                        tx,
                                        &old_entity,
                                        &new_entity,
                                    )
                                    .map_err(|e| {
                                        Error::Execution(format!(
                                            "property index update failed: {e}"
                                        ))
                                    })?;

                                    // For collections: update vectors
                                    if let Some(ref coll) = collection {
                                        if let Some(provider) = ctx.collection_vector_provider() {
                                            for (vector_name, vector_data) in &vectors_to_store {
                                                provider
                                                    .upsert_vector(
                                                        coll.id(),
                                                        new_entity.id,
                                                        table,
                                                        vector_name,
                                                        vector_data,
                                                    )
                                                    .map_err(|e| {
                                                        Error::Execution(format!(
                                                            "vector storage failed: {e}"
                                                        ))
                                                    })?;
                                            }
                                        }
                                    } else {
                                        crate::vector::update_entity_in_indexes(
                                            tx,
                                            &new_entity,
                                            None,
                                        )
                                        .map_err(|e| {
                                            Error::Execution(format!(
                                                "vector index update failed: {e}"
                                            ))
                                        })?;
                                    }

                                    count += 1;
                                }
                            }
                            continue;
                        }
                    },
                    None => {
                        // No ON CONFLICT - this shouldn't happen due to the if let above
                    }
                }
            }

            // No conflict - proceed with normal insert

            // Validate constraints before inserting
            super::constraints::ConstraintValidator::validate_insert(tx, table, &row_values, ctx)
                .map_err(|e| Error::Execution(e.to_string()))?;

            let mut entity = tx.create_entity().map_err(Error::Transaction)?;
            entity = entity.with_label(table);

            // Set properties from row values
            for (col, value) in row_values {
                entity = entity.with_property(&col, value);
            }

            tx.put_entity(&entity).map_err(Error::Transaction)?;

            // Update property indexes for this entity
            super::index_maintenance::EntityIndexMaintenance::on_insert(tx, &entity)
                .map_err(|e| Error::Execution(format!("property index update failed: {e}")))?;

            // For collections: store vectors via CollectionVectorProvider
            if let Some(ref coll) = collection {
                if let Some(provider) = ctx.collection_vector_provider() {
                    for (vector_name, vector_data) in vectors_to_store {
                        provider
                            .upsert_vector(coll.id(), entity.id, table, &vector_name, &vector_data)
                            .map_err(|e| Error::Execution(format!("vector storage failed: {e}")))?;
                    }
                }
            } else {
                // For regular tables: update HNSW indexes via legacy mechanism
                crate::vector::update_entity_in_indexes(tx, &entity, None)
                    .map_err(|e| Error::Execution(format!("vector index update failed: {e}")))?;
            }

            count += 1;
        }
    }

    Ok(count)
}

/// Find an existing entity that conflicts with the row being inserted.
///
/// Returns (Some(entity), true) if a conflict is found, (None, false) otherwise.
fn find_conflicting_entity<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    table: &str,
    target: &manifoldb_query::plan::logical::LogicalConflictTarget,
    row_values: &HashMap<String, Value>,
) -> Result<(Option<Entity>, bool)> {
    use manifoldb_query::plan::logical::LogicalConflictTarget;

    match target {
        LogicalConflictTarget::Columns(columns) => {
            // Find an entity where all conflict columns match
            let entities = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

            for entity in entities {
                let mut all_match = true;

                for col in columns {
                    let existing_value = entity.properties.get(col);
                    let new_value = row_values.get(col);

                    match (existing_value, new_value) {
                        (Some(existing), Some(new)) if existing == new => {
                            // Values match, continue checking
                        }
                        (None, None) => {
                            // Both null, considered a match
                        }
                        _ => {
                            // Values don't match
                            all_match = false;
                            break;
                        }
                    }
                }

                if all_match {
                    return Ok((Some(entity), true));
                }
            }

            Ok((None, false))
        }
        LogicalConflictTarget::Constraint(_constraint_name) => {
            // For constraint-based conflict detection, we would need to look up
            // the constraint definition and find the associated columns.
            // For now, return an error indicating this is not yet supported.
            Err(Error::Execution(
                "ON CONFLICT with constraint name is not yet supported. Use column list instead."
                    .to_string(),
            ))
        }
    }
}

/// Convert a Value to VectorData if it's a vector type.
fn value_to_vector_data(value: &Value) -> Option<manifoldb_vector::types::VectorData> {
    use manifoldb_vector::types::VectorData;

    match value {
        Value::Vector(v) => Some(VectorData::Dense(v.clone())),
        Value::SparseVector(v) => Some(VectorData::Sparse(v.clone())),
        Value::MultiVector(v) => Some(VectorData::Multi(v.clone())),
        _ => None,
    }
}

/// Execute an UPDATE statement.
///
/// For UPDATE ... FROM, the source plan provides rows that are joined with the target
/// table using the WHERE clause. The update only affects target rows that match the join.
fn execute_update<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    assignments: &[(String, LogicalExpr)],
    source: &Option<Box<LogicalPlan>>,
    filter: &Option<LogicalExpr>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};
    use std::collections::HashSet;

    // Check if this is a collection with named vectors
    let collection = CollectionName::new(table)
        .ok()
        .and_then(|name| CollectionManager::get(tx, &name).ok().flatten());

    // Get all entities with this label
    let target_entities: Vec<Entity> = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

    // For UPDATE ... FROM, execute source and find matching target rows
    let (matched_target_ids, source_rows): (HashSet<manifoldb_core::EntityId>, Vec<Entity>) =
        if let Some(source_plan) = source {
            // Execute the source plan
            let source_entities = execute_logical_plan(tx, source_plan, ctx)?;

            // Find target entities that match the join condition (WHERE clause)
            let mut matched_ids = HashSet::new();
            for target in &target_entities {
                for source_entity in &source_entities {
                    // Create a merged entity for evaluating the WHERE clause
                    // Properties are added with both simple and qualified names
                    let mut merged = target.clone();

                    // Add target properties with table-qualified names (e.g., "orders.customer_id")
                    for (key, value) in &target.properties {
                        merged.properties.insert(format!("{table}.{key}"), value.clone());
                    }

                    // Add source properties (both simple and qualified if source has label)
                    for (key, value) in &source_entity.properties {
                        // Add with simple name
                        merged.properties.insert(key.clone(), value.clone());
                        // Add with qualified name if source has labels
                        if let Some(first_label) = source_entity.labels.first() {
                            merged
                                .properties
                                .insert(format!("{}.{key}", first_label.as_str()), value.clone());
                        }
                    }

                    // Check if the join condition (WHERE clause) matches
                    let matches = match filter {
                        Some(pred) => evaluate_predicate(pred, &merged, ctx),
                        None => true, // If no WHERE clause, match all (cartesian product behavior)
                    };

                    if matches {
                        matched_ids.insert(target.id);
                        break; // Only need to match once per target
                    }
                }
            }
            (matched_ids, source_entities)
        } else {
            // Simple UPDATE without FROM - match based on filter only
            let mut matched_ids = HashSet::new();
            for entity in &target_entities {
                let matches = match filter {
                    Some(pred) => evaluate_predicate(pred, entity, ctx),
                    None => true,
                };
                if matches {
                    matched_ids.insert(entity.id);
                }
            }
            (matched_ids, Vec::new())
        };

    let mut count = 0;

    for entity in target_entities {
        if !matched_target_ids.contains(&entity.id) {
            continue;
        }

        // Clone the old entity before modifying
        let old_entity = entity.clone();
        let mut updated_entity = entity;

        // Collect vectors to update separately (for collections)
        let mut vectors_to_update: Vec<(String, manifoldb_vector::types::VectorData)> = Vec::new();

        // For UPDATE FROM, create a merged context with source columns
        // For expression evaluation, we need to provide source row values
        let eval_entity = if source_rows.is_empty() {
            updated_entity.clone()
        } else {
            // Find matching source row for this target
            let mut merged = updated_entity.clone();

            // Add target properties with table-qualified names
            for (key, value) in &updated_entity.properties {
                merged.properties.insert(format!("{table}.{key}"), value.clone());
            }

            for source_entity in &source_rows {
                // Create a test merge to check if this source row matches
                let mut test_merged = merged.clone();
                for (key, value) in &source_entity.properties {
                    test_merged.properties.insert(key.clone(), value.clone());
                    if let Some(first_label) = source_entity.labels.first() {
                        test_merged
                            .properties
                            .insert(format!("{}.{key}", first_label.as_str()), value.clone());
                    }
                }

                // Check if the join condition matches
                let matches = match filter {
                    Some(pred) => evaluate_predicate(pred, &test_merged, ctx),
                    None => true,
                };

                if matches {
                    // Use this merged entity for assignments
                    merged = test_merged;
                    break;
                }
            }
            merged
        };

        // Apply assignments and collect new values for constraint validation
        let mut new_values: HashMap<String, Value> = HashMap::new();
        for (col, expr) in assignments {
            let value = evaluate_expr(expr, &eval_entity, ctx);

            // For collections, check if this column is a named vector
            if let Some(ref coll) = collection {
                if coll.has_vector(col) {
                    // Convert Value to VectorData and update separately
                    if let Some(vector_data) = value_to_vector_data(&value) {
                        vectors_to_update.push((col.clone(), vector_data));
                    }
                    // Remove from entity properties (if it was there)
                    updated_entity.properties.remove(col);
                    continue;
                }
            }

            new_values.insert(col.clone(), value.clone());
            updated_entity.set_property(col, value);
        }

        // Validate constraints for the updated values
        super::constraints::ConstraintValidator::validate_update(
            tx,
            table,
            &old_entity,
            &new_values,
            ctx,
        )
        .map_err(|e| Error::Execution(e.to_string()))?;

        tx.put_entity(&updated_entity).map_err(Error::Transaction)?;

        // Update property indexes for this entity
        super::index_maintenance::EntityIndexMaintenance::on_update(
            tx,
            &old_entity,
            &updated_entity,
        )
        .map_err(|e| Error::Execution(format!("property index update failed: {e}")))?;

        // For collections: update vectors via CollectionVectorProvider
        if let Some(ref coll) = collection {
            if let Some(provider) = ctx.collection_vector_provider() {
                for (vector_name, vector_data) in vectors_to_update {
                    provider
                        .upsert_vector(
                            coll.id(),
                            updated_entity.id,
                            table,
                            &vector_name,
                            &vector_data,
                        )
                        .map_err(|e| Error::Execution(format!("vector storage failed: {e}")))?;
                }
            }
        } else {
            // For regular tables: update HNSW indexes via legacy mechanism
            crate::vector::update_entity_in_indexes(tx, &updated_entity, Some(&old_entity))
                .map_err(|e| Error::Execution(format!("vector index update failed: {e}")))?;
        }

        count += 1;
    }

    Ok(count)
}

/// Execute a DELETE statement.
///
/// For DELETE ... USING, the source plan provides rows that are joined with the target
/// table using the WHERE clause. The delete only affects target rows that match the join.
fn execute_delete<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    source: &Option<Box<LogicalPlan>>,
    filter: &Option<LogicalExpr>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};
    use std::collections::HashSet;

    // Check if this is a collection with named vectors
    let collection = CollectionName::new(table)
        .ok()
        .and_then(|name| CollectionManager::get(tx, &name).ok().flatten());

    // Get all entities with this label
    let target_entities: Vec<Entity> = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

    // For DELETE ... USING, execute source and find matching target rows
    let matched_target_ids: HashSet<manifoldb_core::EntityId> = if let Some(source_plan) = source {
        // Execute the source plan
        let source_entities = execute_logical_plan(tx, source_plan, ctx)?;

        // Find target entities that match the join condition (WHERE clause)
        let mut matched_ids = HashSet::new();
        for target in &target_entities {
            for source_entity in &source_entities {
                // Create a merged entity for evaluating the WHERE clause
                // Properties are added with both simple and qualified names
                let mut merged = target.clone();

                // Add target properties with table-qualified names (e.g., "orders.customer_id")
                for (key, value) in &target.properties {
                    merged.properties.insert(format!("{table}.{key}"), value.clone());
                }

                // Add source properties (both simple and qualified if source has labels)
                for (key, value) in &source_entity.properties {
                    // Add with simple name
                    merged.properties.insert(key.clone(), value.clone());
                    // Add with qualified name if source has labels
                    if let Some(first_label) = source_entity.labels.first() {
                        merged
                            .properties
                            .insert(format!("{}.{key}", first_label.as_str()), value.clone());
                    }
                }

                // Check if the join condition (WHERE clause) matches
                let matches = match filter {
                    Some(pred) => evaluate_predicate(pred, &merged, ctx),
                    None => true, // If no WHERE clause, match all (cartesian product behavior)
                };

                if matches {
                    matched_ids.insert(target.id);
                    break; // Only need to match once per target
                }
            }
        }
        matched_ids
    } else {
        // Simple DELETE without USING - match based on filter only
        let mut matched_ids = HashSet::new();
        for entity in &target_entities {
            let matches = match filter {
                Some(pred) => evaluate_predicate(pred, entity, ctx),
                None => true,
            };
            if matches {
                matched_ids.insert(entity.id);
            }
        }
        matched_ids
    };

    let mut count = 0;

    for entity in target_entities {
        if !matched_target_ids.contains(&entity.id) {
            continue;
        }

        // Validate foreign key constraints before deleting
        // (check that no other table references this row)
        super::constraints::ConstraintValidator::validate_delete(tx, table, &entity)
            .map_err(|e| Error::Execution(e.to_string()))?;

        // Remove from property indexes before deleting
        super::index_maintenance::EntityIndexMaintenance::on_delete(tx, &entity)
            .map_err(|e| Error::Execution(format!("property index removal failed: {e}")))?;

        // For collections: delete all vectors via CollectionVectorProvider
        if let Some(ref coll) = collection {
            if let Some(provider) = ctx.collection_vector_provider() {
                provider
                    .delete_entity_vectors(coll.id(), entity.id, table)
                    .map_err(|e| Error::Execution(format!("vector deletion failed: {e}")))?;
            }
        } else {
            // For regular tables: remove from HNSW indexes before deleting
            crate::vector::remove_entity_from_indexes(tx, &entity)
                .map_err(|e| Error::Execution(format!("vector index removal failed: {e}")))?;
        }

        tx.delete_entity(entity.id).map_err(Error::Transaction)?;
        count += 1;
    }

    Ok(count)
}

/// Execute a SQL MERGE statement.
///
/// MERGE performs conditional INSERT, UPDATE, or DELETE operations based on
/// whether rows from the source match rows in the target table.
///
/// # Algorithm
///
/// 1. Execute the source query to get source rows
/// 2. Get all target entities
/// 3. For each source row, check if it matches any target row (ON condition)
/// 4. Apply WHEN MATCHED clauses to matched pairs
/// 5. Apply WHEN NOT MATCHED clauses to unmatched source rows
/// 6. Apply WHEN NOT MATCHED BY SOURCE clauses to unmatched target rows
fn execute_merge_sql<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    target_table: &str,
    source: &LogicalPlan,
    on_condition: &LogicalExpr,
    clauses: &[LogicalMergeClause],
    ctx: &ExecutionContext,
) -> Result<u64> {
    use std::collections::HashSet;

    // Check if target is a collection with named vectors
    use crate::collection::{CollectionManager, CollectionName};
    let collection = CollectionName::new(target_table)
        .ok()
        .and_then(|name| CollectionManager::get(tx, &name).ok().flatten());

    // Execute the source plan to get source entities
    let source_entities = execute_logical_plan(tx, source, ctx)?;

    // Get all target entities - collect into a Vec so we can iterate multiple times
    let target_entities: Vec<Entity> =
        tx.iter_entities(Some(target_table)).map_err(Error::Transaction)?;

    // Track which target entities have been matched (and potentially modified)
    let mut matched_target_ids: HashSet<manifoldb_core::EntityId> = HashSet::new();

    let mut count = 0u64;

    // Process each source row
    for source_entity in &source_entities {
        // Create a merged entity for evaluating the ON condition
        // This combines source and target properties for condition evaluation
        let mut found_match = false;

        for target_entity in &target_entities {
            // Create a combined entity for evaluating conditions
            // Uses both table names as prefixes for qualified column access
            let merged_entity =
                create_merged_entity_with_aliases(source_entity, target_entity, target_table);

            // Evaluate the ON condition
            if evaluate_predicate(on_condition, &merged_entity, ctx) {
                found_match = true;
                matched_target_ids.insert(target_entity.id);

                // Find the first WHEN MATCHED clause that applies
                for clause in clauses {
                    if clause.match_type != LogicalMergeMatchType::Matched {
                        continue;
                    }

                    // Check additional condition if present
                    let condition_met = match &clause.condition {
                        Some(cond) => evaluate_predicate(cond, &merged_entity, ctx),
                        None => true,
                    };

                    if condition_met {
                        // Execute the action
                        match &clause.action {
                            LogicalMergeAction::Update { assignments } => {
                                let old_entity = target_entity.clone();
                                let mut updated_entity = target_entity.clone();

                                // Collect vectors to update separately (for collections)
                                let mut vectors_to_update: Vec<(
                                    String,
                                    manifoldb_vector::types::VectorData,
                                )> = Vec::new();

                                for (col, expr) in assignments {
                                    // Evaluate expression in context of merged entity
                                    let value = evaluate_expr(expr, &merged_entity, ctx);

                                    // For collections, check if this column is a named vector
                                    if let Some(ref coll) = collection {
                                        if coll.has_vector(col) {
                                            if let Some(vector_data) = value_to_vector_data(&value)
                                            {
                                                vectors_to_update.push((col.clone(), vector_data));
                                            }
                                            updated_entity.properties.remove(col);
                                            continue;
                                        }
                                    }

                                    updated_entity.set_property(col, value);
                                }

                                tx.put_entity(&updated_entity).map_err(Error::Transaction)?;

                                // Update property indexes
                                super::index_maintenance::EntityIndexMaintenance::on_update(
                                    tx,
                                    &old_entity,
                                    &updated_entity,
                                )
                                .map_err(|e| {
                                    Error::Execution(format!("property index update failed: {e}"))
                                })?;

                                // Update vectors for collections
                                if let Some(ref coll) = collection {
                                    if let Some(provider) = ctx.collection_vector_provider() {
                                        for (vector_name, vector_data) in vectors_to_update {
                                            provider
                                                .upsert_vector(
                                                    coll.id(),
                                                    updated_entity.id,
                                                    target_table,
                                                    &vector_name,
                                                    &vector_data,
                                                )
                                                .map_err(|e| {
                                                    Error::Execution(format!(
                                                        "vector storage failed: {e}"
                                                    ))
                                                })?;
                                        }
                                    }
                                } else {
                                    // For regular tables: update HNSW indexes
                                    crate::vector::update_entity_in_indexes(
                                        tx,
                                        &updated_entity,
                                        Some(&old_entity),
                                    )
                                    .map_err(|e| {
                                        Error::Execution(format!("vector index update failed: {e}"))
                                    })?;
                                }

                                count += 1;
                            }
                            LogicalMergeAction::Delete => {
                                // Remove from property indexes
                                super::index_maintenance::EntityIndexMaintenance::on_delete(
                                    tx,
                                    target_entity,
                                )
                                .map_err(|e| {
                                    Error::Execution(format!("property index removal failed: {e}"))
                                })?;

                                // Delete vectors for collections
                                if let Some(ref coll) = collection {
                                    if let Some(provider) = ctx.collection_vector_provider() {
                                        provider
                                            .delete_entity_vectors(
                                                coll.id(),
                                                target_entity.id,
                                                target_table,
                                            )
                                            .map_err(|e| {
                                                Error::Execution(format!(
                                                    "vector deletion failed: {e}"
                                                ))
                                            })?;
                                    }
                                } else {
                                    crate::vector::remove_entity_from_indexes(tx, target_entity)
                                        .map_err(|e| {
                                            Error::Execution(format!(
                                                "vector index removal failed: {e}"
                                            ))
                                        })?;
                                }

                                tx.delete_entity(target_entity.id).map_err(Error::Transaction)?;
                                count += 1;
                            }
                            LogicalMergeAction::DoNothing => {
                                // Do nothing, but count as processed
                            }
                            LogicalMergeAction::Insert { .. } => {
                                // INSERT is not valid for WHEN MATCHED
                                return Err(Error::Execution(
                                    "INSERT action is not valid for WHEN MATCHED clause"
                                        .to_string(),
                                ));
                            }
                        }

                        // Only apply the first matching clause
                        break;
                    }
                }

                // Only process the first matching target row per source row
                break;
            }
        }

        // If no match found, apply WHEN NOT MATCHED clauses
        if !found_match {
            for clause in clauses {
                if clause.match_type != LogicalMergeMatchType::NotMatched {
                    continue;
                }

                // Check additional condition if present
                let condition_met = match &clause.condition {
                    Some(cond) => evaluate_predicate(cond, source_entity, ctx),
                    None => true,
                };

                if condition_met {
                    match &clause.action {
                        LogicalMergeAction::Insert { columns, values } => {
                            // Create a new entity with a new ID from the transaction
                            let mut new_entity = tx.create_entity().map_err(Error::Transaction)?;
                            new_entity = new_entity.with_label(target_table);

                            // Collect vectors to store separately (for collections)
                            let mut vectors_to_store: Vec<(
                                String,
                                manifoldb_vector::types::VectorData,
                            )> = Vec::new();

                            for (col, expr) in columns.iter().zip(values.iter()) {
                                let value = evaluate_expr(expr, source_entity, ctx);

                                // For collections, check if this column is a named vector
                                if let Some(ref coll) = collection {
                                    if coll.has_vector(col) {
                                        if let Some(vector_data) = value_to_vector_data(&value) {
                                            vectors_to_store.push((col.clone(), vector_data));
                                        }
                                        continue;
                                    }
                                }

                                new_entity = new_entity.with_property(col, value);
                            }

                            tx.put_entity(&new_entity).map_err(Error::Transaction)?;

                            // Update property indexes
                            super::index_maintenance::EntityIndexMaintenance::on_insert(
                                tx,
                                &new_entity,
                            )
                            .map_err(|e| {
                                Error::Execution(format!("property index insert failed: {e}"))
                            })?;

                            // Store vectors for collections
                            if let Some(ref coll) = collection {
                                if let Some(provider) = ctx.collection_vector_provider() {
                                    for (vector_name, vector_data) in vectors_to_store {
                                        provider
                                            .upsert_vector(
                                                coll.id(),
                                                new_entity.id,
                                                target_table,
                                                &vector_name,
                                                &vector_data,
                                            )
                                            .map_err(|e| {
                                                Error::Execution(format!(
                                                    "vector storage failed: {e}"
                                                ))
                                            })?;
                                    }
                                }
                            } else {
                                crate::vector::update_entity_in_indexes(tx, &new_entity, None)
                                    .map_err(|e| {
                                        Error::Execution(format!("vector index insert failed: {e}"))
                                    })?;
                            }

                            count += 1;
                        }
                        LogicalMergeAction::DoNothing => {
                            // Do nothing
                        }
                        _ => {
                            return Err(Error::Execution(
                                "Only INSERT or DO NOTHING is valid for WHEN NOT MATCHED clause"
                                    .to_string(),
                            ));
                        }
                    }

                    // Only apply the first matching clause
                    break;
                }
            }
        }
    }

    // Process WHEN NOT MATCHED BY SOURCE clauses (target rows not in source)
    let has_not_matched_by_source =
        clauses.iter().any(|c| c.match_type == LogicalMergeMatchType::NotMatchedBySource);

    if has_not_matched_by_source {
        for target_entity in &target_entities {
            if matched_target_ids.contains(&target_entity.id) {
                continue; // Already matched, skip
            }

            for clause in clauses {
                if clause.match_type != LogicalMergeMatchType::NotMatchedBySource {
                    continue;
                }

                // Check additional condition if present
                let condition_met = match &clause.condition {
                    Some(cond) => evaluate_predicate(cond, target_entity, ctx),
                    None => true,
                };

                if condition_met {
                    match &clause.action {
                        LogicalMergeAction::Update { assignments } => {
                            let old_entity = target_entity.clone();
                            let mut updated_entity = target_entity.clone();

                            let mut vectors_to_update: Vec<(
                                String,
                                manifoldb_vector::types::VectorData,
                            )> = Vec::new();

                            for (col, expr) in assignments {
                                let value = evaluate_expr(expr, target_entity, ctx);

                                if let Some(ref coll) = collection {
                                    if coll.has_vector(col) {
                                        if let Some(vector_data) = value_to_vector_data(&value) {
                                            vectors_to_update.push((col.clone(), vector_data));
                                        }
                                        updated_entity.properties.remove(col);
                                        continue;
                                    }
                                }

                                updated_entity.set_property(col, value);
                            }

                            tx.put_entity(&updated_entity).map_err(Error::Transaction)?;

                            super::index_maintenance::EntityIndexMaintenance::on_update(
                                tx,
                                &old_entity,
                                &updated_entity,
                            )
                            .map_err(|e| {
                                Error::Execution(format!("property index update failed: {e}"))
                            })?;

                            if let Some(ref coll) = collection {
                                if let Some(provider) = ctx.collection_vector_provider() {
                                    for (vector_name, vector_data) in vectors_to_update {
                                        provider
                                            .upsert_vector(
                                                coll.id(),
                                                updated_entity.id,
                                                target_table,
                                                &vector_name,
                                                &vector_data,
                                            )
                                            .map_err(|e| {
                                                Error::Execution(format!(
                                                    "vector storage failed: {e}"
                                                ))
                                            })?;
                                    }
                                }
                            } else {
                                crate::vector::update_entity_in_indexes(
                                    tx,
                                    &updated_entity,
                                    Some(&old_entity),
                                )
                                .map_err(|e| {
                                    Error::Execution(format!("vector index update failed: {e}"))
                                })?;
                            }

                            count += 1;
                        }
                        LogicalMergeAction::Delete => {
                            super::index_maintenance::EntityIndexMaintenance::on_delete(
                                tx,
                                target_entity,
                            )
                            .map_err(|e| {
                                Error::Execution(format!("property index removal failed: {e}"))
                            })?;

                            if let Some(ref coll) = collection {
                                if let Some(provider) = ctx.collection_vector_provider() {
                                    provider
                                        .delete_entity_vectors(
                                            coll.id(),
                                            target_entity.id,
                                            target_table,
                                        )
                                        .map_err(|e| {
                                            Error::Execution(format!("vector deletion failed: {e}"))
                                        })?;
                                }
                            } else {
                                crate::vector::remove_entity_from_indexes(tx, target_entity)
                                    .map_err(|e| {
                                        Error::Execution(format!(
                                            "vector index removal failed: {e}"
                                        ))
                                    })?;
                            }

                            tx.delete_entity(target_entity.id).map_err(Error::Transaction)?;
                            count += 1;
                        }
                        LogicalMergeAction::DoNothing => {
                            // Do nothing
                        }
                        LogicalMergeAction::Insert { .. } => {
                            return Err(Error::Execution(
                                "INSERT is not valid for WHEN NOT MATCHED BY SOURCE clause"
                                    .to_string(),
                            ));
                        }
                    }

                    break;
                }
            }
        }
    }

    Ok(count)
}

/// Create a merged entity combining source and target for condition evaluation.
///
/// This is used in MERGE statements to evaluate the ON condition and WHEN clause
/// conditions where both source and target columns may be referenced.
///
/// The merged entity contains:
/// - All target properties under their original names and with "target." prefix
/// - All source properties under their original names (if no collision) and with "source." prefix
fn create_merged_entity_with_aliases(
    source: &Entity,
    target: &Entity,
    target_table: &str,
) -> Entity {
    let mut merged = target.clone();

    // Add target properties with "target." prefix for qualified access (e.g., target.id)
    for (key, value) in &target.properties {
        merged.properties.insert(format!("target.{}", key), value.clone());
        merged.properties.insert(format!("{}.{}", target_table, key), value.clone());
    }

    // Add source properties with "source." prefix for qualified access
    // Also add without prefix if no collision with target
    for (key, value) in &source.properties {
        merged.properties.insert(format!("source.{}", key), value.clone());
        // Don't override target properties - source is secondary context for unqualified names
        if !merged.properties.contains_key(key) {
            merged.properties.insert(key.clone(), value.clone());
        }
    }

    merged
}

/// Check if a logical plan contains a JOIN node.
fn contains_join(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Join { .. } => true,
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Distinct { input, .. }
        | LogicalPlan::Alias { input, .. }
        | LogicalPlan::Aggregate { input, .. } => contains_join(input),
        _ => false,
    }
}

/// Check if a logical plan contains a graph traversal node (Expand or PathScan).
fn contains_graph(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Expand { .. } | LogicalPlan::PathScan { .. } => true,
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Distinct { input, .. }
        | LogicalPlan::Alias { input, .. }
        | LogicalPlan::Aggregate { input, .. } => contains_graph(input),
        _ => false,
    }
}

/// Execute a JOIN query and return the result set.
fn execute_join<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: &manifoldb_query::plan::logical::JoinNode,
    left: &LogicalPlan,
    right: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // Execute both sides to get entities
    let (left_entities, left_alias) = execute_join_input(tx, left, ctx)?;
    let (right_entities, right_alias) = execute_join_input(tx, right, ctx)?;

    // Create ValuesOp operators for both sides with proper column prefixes
    let (left_op, left_cols) = entities_to_values_op(&left_entities, &left_alias);
    let (right_op, right_cols) = entities_to_values_op(&right_entities, &right_alias);

    // Determine the appropriate join operator
    let mut join_op = create_join_operator(
        node.join_type,
        node.condition.clone(),
        &node.using_columns,
        Box::new(left_op),
        Box::new(right_op),
        &left_cols,
        &right_cols,
    );

    // Execute the join
    join_op.open(ctx).map_err(|e| Error::Execution(e.to_string()))?;

    let mut rows = Vec::new();
    while let Some(row) = join_op.next().map_err(|e| Error::Execution(e.to_string()))? {
        rows.push(row);
    }

    join_op.close().map_err(|e| Error::Execution(e.to_string()))?;

    let schema = join_op.schema();
    Ok(ResultSet::with_rows(schema, rows))
}

/// Execute a projection over a JOIN result.
fn execute_join_projection<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    exprs: &[LogicalExpr],
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // First execute the join to get rows
    let join_result = execute_join_plan(tx, input, ctx)?;

    // Check for wildcard projection
    let has_wildcard = exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

    if has_wildcard {
        // Return all columns from the join
        Ok(join_result)
    } else {
        // Project specific columns
        let projected_columns: Vec<String> = exprs.iter().map(|e| expr_to_column_name(e)).collect();
        let new_schema = Arc::new(Schema::new(projected_columns.clone()));

        let mut projected_rows = Vec::new();
        for row in join_result.rows() {
            let values: Vec<Value> =
                exprs.iter().map(|expr| evaluate_row_expr(expr, row)).collect();
            projected_rows.push(Row::new(Arc::clone(&new_schema), values));
        }

        Ok(ResultSet::with_rows(new_schema, projected_rows))
    }
}

/// Execute a plan that may contain a JOIN, returning a ResultSet.
fn execute_join_plan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    match plan {
        LogicalPlan::Join { node, left, right } => execute_join(tx, node, left, right, ctx),
        LogicalPlan::Filter { node, input } => {
            // Apply filter to join result
            let join_result = execute_join_plan(tx, input, ctx)?;
            let schema = join_result.schema_arc();
            let filtered_rows: Vec<Row> = join_result
                .into_rows()
                .into_iter()
                .filter(|row| {
                    let result = evaluate_row_expr(&node.predicate, row);
                    matches!(result, Value::Bool(true))
                })
                .collect();
            Ok(ResultSet::with_rows(schema, filtered_rows))
        }
        LogicalPlan::Sort { node, input } => {
            // Apply sort to join result
            let join_result = execute_join_plan(tx, input, ctx)?;
            let schema = join_result.schema_arc();
            let mut rows = join_result.into_rows();

            if !node.order_by.is_empty() {
                rows.sort_by(|a, b| {
                    // Compare by each sort key in order
                    for order in &node.order_by {
                        let va = evaluate_row_expr(&order.expr, a);
                        let vb = evaluate_row_expr(&order.expr, b);
                        let cmp = compare_values(&va, &vb);
                        let cmp = if order.ascending { cmp } else { cmp.reverse() };
                        if cmp != std::cmp::Ordering::Equal {
                            return cmp;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }

            Ok(ResultSet::with_rows(schema, rows))
        }
        LogicalPlan::Limit { node, input } => {
            // Apply limit to join result
            let join_result = execute_join_plan(tx, input, ctx)?;
            let schema = join_result.schema_arc();
            let rows = join_result.into_rows();

            let start = node.offset.unwrap_or(0);
            let end = node.limit.map(|l| start + l).unwrap_or(rows.len());
            let limited_rows: Vec<Row> = rows.into_iter().skip(start).take(end - start).collect();

            Ok(ResultSet::with_rows(schema, limited_rows))
        }
        LogicalPlan::Alias { input, .. } => {
            // Alias doesn't change the rows
            execute_join_plan(tx, input, ctx)
        }
        _ => {
            // Fall back to entity execution for non-join plans
            let entities = execute_logical_plan(tx, plan, ctx)?;
            let columns = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, columns);
            let schema = scan.schema();
            let rows = scan.collect_rows();
            Ok(ResultSet::with_rows(schema, rows))
        }
    }
}

/// Execute a join input and return entities along with the table alias.
fn execute_join_input<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<(Vec<Entity>, String)> {
    match plan {
        LogicalPlan::Scan(scan_node) => {
            let label = &scan_node.table_name;
            let alias = scan_node.alias.as_deref().unwrap_or(label);

            // Check if this is a materialized view
            if let Some(entities) = try_execute_materialized_view_as_entities(tx, label)? {
                return Ok((entities, alias.to_string()));
            }

            let entities = tx.iter_entities(Some(label)).map_err(Error::Transaction)?;
            Ok((entities, alias.to_string()))
        }
        LogicalPlan::Alias { alias, input } => {
            let (entities, _) = execute_join_input(tx, input, ctx)?;
            Ok((entities, alias.clone()))
        }
        LogicalPlan::Filter { node, input } => {
            let (entities, alias) = execute_join_input(tx, input, ctx)?;
            let predicate = &node.predicate;
            let filtered: Vec<Entity> = entities
                .into_iter()
                .filter(|entity| evaluate_predicate_with_tx(tx, predicate, entity, ctx))
                .collect();
            Ok((filtered, alias))
        }
        LogicalPlan::Join { node, left, right } => {
            // Nested join - execute recursively as rows, then we need to handle differently
            // For now, execute inner joins and convert to a synthetic entity representation
            let result = execute_join(tx, node, left, right, ctx)?;

            // Convert rows back to synthetic entities for the outer join
            let entities: Vec<Entity> = result
                .rows()
                .iter()
                .enumerate()
                .map(|(i, row)| row_to_entity(row, i as u64))
                .collect();

            Ok((entities, "joined".to_string()))
        }
        _ => {
            // Execute other plan types normally
            let entities = execute_logical_plan(tx, plan, ctx)?;
            Ok((entities, "table".to_string()))
        }
    }
}

/// Convert entities to a ValuesOp with prefixed column names.
fn entities_to_values_op(entities: &[Entity], prefix: &str) -> (ValuesOp, Vec<String>) {
    // Collect all unique property names
    let mut prop_names: Vec<String> = vec!["_rowid".to_string()];
    for entity in entities {
        for key in entity.properties.keys() {
            if !prop_names.contains(key) {
                prop_names.push(key.clone());
            }
        }
    }

    // Create prefixed column names (e.g., "u._rowid", "u.name")
    let prefixed_columns: Vec<String> =
        prop_names.iter().map(|n| format!("{}.{}", prefix, n)).collect();

    // Convert entities to row values
    let rows: Vec<Vec<Value>> = entities
        .iter()
        .map(|entity| {
            prop_names
                .iter()
                .map(|prop| {
                    if prop == "_rowid" {
                        Value::Int(entity.id.as_u64() as i64)
                    } else {
                        entity.get_property(prop).cloned().unwrap_or(Value::Null)
                    }
                })
                .collect()
        })
        .collect();

    (ValuesOp::with_columns(prefixed_columns.clone(), rows), prefixed_columns)
}

/// Create the appropriate join operator based on join type and condition.
fn create_join_operator(
    join_type: JoinType,
    condition: Option<LogicalExpr>,
    using_columns: &[String],
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_cols: &[String],
    right_cols: &[String],
) -> Box<dyn Operator> {
    // Handle USING clause by converting to equijoin condition
    let join_condition = if using_columns.is_empty() {
        condition
    } else {
        // Convert USING(col1, col2) to left.col1 = right.col1 AND left.col2 = right.col2
        let conditions: Vec<LogicalExpr> = using_columns
            .iter()
            .filter_map(|col| {
                // Find matching columns in left and right
                let left_col = left_cols.iter().find(|c| c.ends_with(&format!(".{}", col)));
                let right_col = right_cols.iter().find(|c| c.ends_with(&format!(".{}", col)));

                match (left_col, right_col) {
                    (Some(l), Some(r)) => Some(LogicalExpr::column(l).eq(LogicalExpr::column(r))),
                    _ => None,
                }
            })
            .collect();

        if conditions.is_empty() {
            condition
        } else {
            // Combine with AND
            let mut combined = conditions.into_iter();
            let first = combined.next();
            first.map(|f| combined.fold(f, |acc, c| acc.and(c)))
        }
    };

    // Check if we can use a hash join (equijoin on simple columns)
    if let Some(ref cond) = join_condition {
        if let Some((left_keys, right_keys)) = extract_equijoin_keys(cond, left_cols, right_cols) {
            // For LEFT join: probe with left (all left rows appear), build with right
            // For INNER join: probe with right (smaller side typically), build with left
            // HashJoinOp: build_null_row().merge(probe) for unmatched probe rows (supports LEFT)
            return match join_type {
                JoinType::Left => {
                    // Swap: build on right, probe with left for LEFT OUTER join
                    Box::new(HashJoinOp::new(join_type, right_keys, left_keys, None, right, left))
                }
                _ => Box::new(HashJoinOp::new(join_type, left_keys, right_keys, None, left, right)),
            };
        }
    }

    // Fall back to nested loop join
    Box::new(NestedLoopJoinOp::new(join_type, join_condition, left, right))
}

/// Extract equijoin keys from a join condition if possible.
fn extract_equijoin_keys(
    condition: &LogicalExpr,
    left_cols: &[String],
    right_cols: &[String],
) -> Option<(Vec<LogicalExpr>, Vec<LogicalExpr>)> {
    match condition {
        LogicalExpr::BinaryOp { left, op, right } => {
            use manifoldb_query::ast::BinaryOp;
            match op {
                BinaryOp::Eq => {
                    // Check if this is a simple column = column equality
                    let left_col = extract_column_name(left);
                    let right_col = extract_column_name(right);

                    if let (Some(l), Some(r)) = (left_col, right_col) {
                        // Determine which side each column belongs to
                        let l_is_left =
                            left_cols.iter().any(|c| c == l || c.ends_with(&format!(".{}", l)));
                        let r_is_right =
                            right_cols.iter().any(|c| c == r || c.ends_with(&format!(".{}", r)));

                        if l_is_left && r_is_right {
                            return Some((vec![*left.clone()], vec![*right.clone()]));
                        }

                        // Try the other way
                        let l_is_right =
                            right_cols.iter().any(|c| c == l || c.ends_with(&format!(".{}", l)));
                        let r_is_left =
                            left_cols.iter().any(|c| c == r || c.ends_with(&format!(".{}", r)));

                        if l_is_right && r_is_left {
                            return Some((vec![*right.clone()], vec![*left.clone()]));
                        }
                    }
                    None
                }
                BinaryOp::And => {
                    // Try to extract keys from both sides of AND
                    let left_keys = extract_equijoin_keys(left, left_cols, right_cols);
                    let right_keys = extract_equijoin_keys(right, left_cols, right_cols);

                    match (left_keys, right_keys) {
                        (Some((mut lk1, mut rk1)), Some((lk2, rk2))) => {
                            lk1.extend(lk2);
                            rk1.extend(rk2);
                            Some((lk1, rk1))
                        }
                        (Some(keys), None) | (None, Some(keys)) => Some(keys),
                        (None, None) => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract a column name from an expression.
fn extract_column_name(expr: &LogicalExpr) -> Option<&str> {
    match expr {
        LogicalExpr::Column { name, .. } => Some(name.as_str()),
        LogicalExpr::Alias { expr, .. } => extract_column_name(expr),
        _ => None,
    }
}

/// Convert a row back to a synthetic entity (for nested joins).
fn row_to_entity(row: &Row, id: u64) -> Entity {
    let mut entity = Entity::new(manifoldb_core::EntityId::new(id));

    for (i, col) in row.schema().columns().iter().enumerate() {
        if let Some(value) = row.get(i) {
            entity.set_property(*col, value.clone());
        }
    }

    entity
}

/// Evaluate a logical expression against a row (for join results).
fn evaluate_row_expr(expr: &LogicalExpr, row: &Row) -> Value {
    use manifoldb_query::exec::operators::filter::evaluate_expr as op_evaluate_expr;

    // Use the operator's evaluate_expr which works with Row
    op_evaluate_expr(expr, row).unwrap_or(Value::Null)
}

/// Evaluate a logical expression against a row with subquery support.
///
/// This version properly handles EXISTS, IN subquery, and scalar subquery expressions
/// by delegating to the filter operator's evaluate_expr_with_subquery function.
///
/// A new minimal execution context is created for subquery evaluation since the
/// subquery functions create their own correlated contexts internally.
fn evaluate_row_expr_with_subquery(
    expr: &LogicalExpr,
    row: &Row,
    base_ctx: &ExecutionContext,
) -> Value {
    use manifoldb_query::exec::operators::filter::evaluate_expr_with_subquery;

    // Create a minimal context for subquery evaluation that preserves graph accessors
    // The subquery functions (evaluate_sql_exists_subquery, etc.) will create their
    // own correlated contexts with outer row bindings
    let subquery_ctx = ExecutionContext::new()
        .with_graph(base_ctx.graph_arc())
        .with_graph_mutator(base_ctx.graph_mutator_arc());

    let ctx_opt: Option<ExecutionContext> = Some(subquery_ctx);

    evaluate_expr_with_subquery(expr, row, None, &ctx_opt).unwrap_or(Value::Null)
}

/// Evaluate a logical expression against a row with transaction-based subquery support.
///
/// This version uses the transaction to execute subqueries against database tables,
/// providing full support for EXISTS, IN, and scalar subqueries that reference tables.
fn evaluate_row_expr_with_subquery_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    expr: &LogicalExpr,
    row: &Row,
    base_ctx: &ExecutionContext,
) -> Value {
    match expr {
        // SQL EXISTS subquery - use transaction to execute against database
        LogicalExpr::Exists { subquery, negated } => {
            let result = evaluate_subquery_exists_with_tx(tx, subquery, base_ctx);
            Value::Bool(if *negated { !result } else { result })
        }

        // SQL IN subquery - use transaction to execute against database
        LogicalExpr::InSubquery { expr: inner_expr, subquery, negated } => {
            let val = evaluate_row_expr_with_subquery_tx(tx, inner_expr, row, base_ctx);
            if matches!(val, Value::Null) {
                return Value::Null;
            }
            let result = evaluate_subquery_in_with_tx(tx, &val, subquery, base_ctx);
            Value::Bool(if *negated { !result } else { result })
        }

        // SQL scalar subquery - use transaction to execute against database
        LogicalExpr::Subquery(subquery) => evaluate_scalar_subquery_with_tx(tx, subquery, base_ctx),

        // Binary operations - recursively handle subqueries in operands
        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_row_expr_with_subquery_tx(tx, left, row, base_ctx);
            let rval = evaluate_row_expr_with_subquery_tx(tx, right, row, base_ctx);
            evaluate_binary_op(op, &lval, &rval)
        }

        // Unary operations
        LogicalExpr::UnaryOp { op, operand } => {
            let val = evaluate_row_expr_with_subquery_tx(tx, operand, row, base_ctx);
            match op {
                manifoldb_query::ast::UnaryOp::Not => match val {
                    Value::Bool(b) => Value::Bool(!b),
                    _ => Value::Null,
                },
                manifoldb_query::ast::UnaryOp::Neg => match val {
                    Value::Int(i) => Value::Int(-i),
                    Value::Float(f) => Value::Float(-f),
                    _ => Value::Null,
                },
                manifoldb_query::ast::UnaryOp::IsNull => Value::Bool(matches!(val, Value::Null)),
                manifoldb_query::ast::UnaryOp::IsNotNull => {
                    Value::Bool(!matches!(val, Value::Null))
                }
            }
        }

        // Column reference - look up in the row
        LogicalExpr::Column { name, qualifier } => {
            let schema = row.schema();
            // Try qualified name first if available
            if let Some(q) = qualifier {
                let qualified_name = format!("{}.{}", q, name);
                if let Some(idx) = schema.index_of(&qualified_name) {
                    return row.get(idx).cloned().unwrap_or(Value::Null);
                }
            }
            // Then try just the column name
            if let Some(idx) = schema.index_of(name) {
                row.get(idx).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        // Literal values
        LogicalExpr::Literal(lit) => match lit {
            Literal::Null => Value::Null,
            Literal::Boolean(b) => Value::Bool(*b),
            Literal::Integer(i) => Value::Int(*i),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
            Literal::Vector(v) => Value::Vector(v.clone()),
            Literal::MultiVector(v) => Value::MultiVector(v.clone()),
        },

        // For other expressions, fall back to the non-tx version
        _ => evaluate_row_expr_with_subquery(expr, row, base_ctx),
    }
}

/// Execute a CREATE TABLE statement.
fn execute_create_table<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateTableNode,
) -> Result<u64> {
    SchemaManager::create_table(tx, node).map_err(|e| Error::Execution(e.to_string()))?;
    Ok(0) // DDL doesn't return row counts
}

/// Execute an ALTER TABLE statement.
fn execute_alter_table<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &AlterTableNode,
) -> Result<u64> {
    SchemaManager::alter_table(tx, node).map_err(|e| Error::Execution(e.to_string()))?;
    Ok(0) // DDL doesn't return row counts
}

/// Execute a DROP TABLE statement.
fn execute_drop_table<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &DropTableNode,
) -> Result<u64> {
    for table_name in &node.names {
        SchemaManager::drop_table(tx, table_name, node.if_exists)
            .map_err(|e| Error::Execution(e.to_string()))?;

        // Also delete all entities with this label if cascade or always
        // For now, we'll delete entities when dropping a table
        let entities = tx.iter_entities(Some(table_name)).map_err(Error::Transaction)?;
        for entity in entities {
            tx.delete_entity(entity.id).map_err(Error::Transaction)?;
        }
    }
    Ok(0)
}

/// Execute a CREATE INDEX statement.
fn execute_create_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateIndexNode,
) -> Result<u64> {
    // Store schema metadata
    SchemaManager::create_index(tx, node).map_err(|e| Error::Execution(e.to_string()))?;

    // Check the index type and build appropriately
    if let Some(using) = &node.using {
        let using_lower = using.to_lowercase();
        if using_lower == "hnsw" {
            build_hnsw_index(tx, node)?;
            return Ok(0);
        }
        // Other index types (btree, hash) fall through to backfill
    }

    // For BTree/Hash indexes (or no USING clause), backfill existing data
    let backfilled = backfill_btree_index(tx, node)?;
    Ok(backfilled)
}

/// Backfill a BTree index with existing entity data.
///
/// This scans all entities with the specified label and creates index entries
/// for the indexed columns.
fn backfill_btree_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateIndexNode,
) -> Result<u64> {
    use manifoldb_core::index::{IndexId, PropertyIndexEntry};

    // Extract column names from the index definition
    let column_names: Vec<String> =
        node.columns.iter().map(|c| extract_column_name_from_expr(&c.expr)).collect();

    if column_names.is_empty() {
        return Ok(0);
    }

    // For now, only support single-column indexes for property index backfill
    if column_names.len() > 1 {
        // Multi-column indexes are stored in schema but not backfilled to property index
        // They can be used for query planning but require different storage format
        return Ok(0);
    }

    let column_name = &column_names[0];
    let table_name = &node.table;

    // Create the index ID for this label + property combination
    let index_id = IndexId::from_label_property(table_name, column_name);

    // Scan all entities with this label and create index entries
    let entities =
        tx.iter_entities(Some(table_name)).map_err(|e| Error::Execution(e.to_string()))?;

    let mut count = 0u64;
    for entity in &entities {
        // Get the property value for this column
        if let Some(value) = entity.properties.get(column_name) {
            // Only index scalar values (not vectors, arrays, etc.)
            if PropertyIndexEntry::is_indexable(value) {
                let entry = PropertyIndexEntry::new(index_id, value.clone(), entity.id);
                if let Some(key) = entry.encode_key() {
                    tx.put_property_index(&key).map_err(|e| Error::Execution(e.to_string()))?;
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}

/// Build an HNSW index from the CREATE INDEX node.
fn build_hnsw_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateIndexNode,
) -> Result<()> {
    use crate::vector::HnswIndexBuilder;
    use manifoldb_vector::distance::DistanceMetric;

    // Extract the column name from the first index column
    let column_name =
        node.columns.first().map(|c| extract_column_name_from_expr(&c.expr)).ok_or_else(|| {
            Error::Execution("HNSW index requires exactly one column".to_string())
        })?;

    // Parse options from WITH clause
    let mut builder = HnswIndexBuilder::new(&node.name, &node.table, column_name);

    for (key, value) in &node.with {
        let key_lower = key.to_lowercase();
        match key_lower.as_str() {
            "m" => {
                if let Ok(m) = value.parse::<usize>() {
                    builder = builder.m(m);
                }
            }
            "ef_construction" => {
                if let Ok(ef) = value.parse::<usize>() {
                    builder = builder.ef_construction(ef);
                }
            }
            "ef_search" => {
                if let Ok(ef) = value.parse::<usize>() {
                    builder = builder.ef_search(ef);
                }
            }
            "dimension" | "dimensions" => {
                if let Ok(dim) = value.parse::<usize>() {
                    builder = builder.dimension(dim);
                }
            }
            "distance" | "metric" | "distance_metric" => {
                let metric = match value.to_lowercase().as_str() {
                    "euclidean" | "l2" => DistanceMetric::Euclidean,
                    "cosine" => DistanceMetric::Cosine,
                    "dot" | "inner_product" | "ip" => DistanceMetric::DotProduct,
                    _ => DistanceMetric::Cosine, // Default
                };
                builder = builder.distance_metric(metric);
            }
            _ => {
                // Ignore unknown options
            }
        }
    }

    builder.build(tx).map_err(|e| Error::Execution(e.to_string()))
}

/// Extract column name from an index column expression.
fn extract_column_name_from_expr(expr: &manifoldb_query::ast::Expr) -> String {
    match expr {
        manifoldb_query::ast::Expr::Column(qn) => {
            qn.parts.last().map(|p| p.name.clone()).unwrap_or_default()
        }
        _ => format!("{expr:?}"),
    }
}

/// Execute a DROP INDEX statement.
fn execute_drop_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &DropIndexNode,
) -> Result<u64> {
    let mut total_deleted = 0u64;

    for index_name in &node.names {
        // Get index schema before deleting to know what to clean up
        let schema = SchemaManager::get_index(tx, index_name)
            .map_err(|e| Error::Execution(e.to_string()))?;

        // Check if this is an HNSW index and drop it
        // Pass if_exists=true since the index might be a non-HNSW index (schema-only)
        crate::vector::drop_index(tx, index_name, true)
            .map_err(|e| Error::Execution(format!("failed to drop vector index: {e}")))?;

        // Clean up property index entries for BTree indexes
        if let Some(schema) = schema {
            let is_hnsw = schema.using.as_ref().is_some_and(|u| u.eq_ignore_ascii_case("hnsw"));
            if !is_hnsw {
                let deleted = cleanup_btree_index_entries(tx, &schema)?;
                total_deleted += deleted;
            }
        }

        // Drop from schema manager
        SchemaManager::drop_index(tx, index_name, node.if_exists)
            .map_err(|e| Error::Execution(e.to_string()))?;
    }
    Ok(total_deleted)
}

/// Execute an ALTER INDEX statement.
fn execute_alter_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &AlterIndexNode,
) -> Result<u64> {
    use manifoldb_query::plan::logical::AlterIndexAction;

    // Check if index exists
    let existing =
        SchemaManager::get_index(tx, &node.name).map_err(|e| Error::Execution(e.to_string()))?;

    let Some(mut schema) = existing else {
        if node.if_exists {
            return Ok(0);
        }
        return Err(Error::Execution(format!("Index '{}' does not exist", node.name)));
    };

    match &node.action {
        AlterIndexAction::RenameIndex { new_name } => {
            // Check the new name doesn't already exist
            let new_exists = SchemaManager::get_index(tx, new_name)
                .map_err(|e| Error::Execution(e.to_string()))?;
            if new_exists.is_some() {
                return Err(Error::Execution(format!("Index '{}' already exists", new_name)));
            }

            // Drop the old index schema entry
            SchemaManager::drop_index(tx, &node.name, false)
                .map_err(|e| Error::Execution(e.to_string()))?;

            // Update the schema with the new name
            schema.name.clone_from(new_name);

            // Store with new name directly using bincode
            let key = format!("schema:index:{}", schema.name);
            let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
                .map_err(|e| Error::Execution(format!("Failed to serialize index schema: {e}")))?;
            tx.put_metadata(key.as_bytes(), &value).map_err(Error::Transaction)?;

            // Update the index list
            let mut indexes = SchemaManager::list_indexes(tx).unwrap_or_default();
            if !indexes.contains(&schema.name) {
                indexes.push(schema.name.clone());
                let list_value =
                    bincode::serde::encode_to_vec(&indexes, bincode::config::standard()).map_err(
                        |e| Error::Execution(format!("Failed to serialize indexes list: {e}")),
                    )?;
                tx.put_metadata(b"schema:indexes_list", &list_value).map_err(Error::Transaction)?;
            }
        }
        AlterIndexAction::SetOptions { options } => {
            // Update options in the schema (with_options is a Vec<(String, String)>)
            for (key, value) in options {
                // Remove existing option with same key, if any
                schema.with_options.retain(|(k, _)| k != key);
                // Add the new option
                schema.with_options.push((key.clone(), value.clone()));
            }
            // Update in storage
            let key = format!("schema:index:{}", schema.name);
            let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
                .map_err(|e| Error::Execution(format!("Failed to serialize index schema: {e}")))?;
            tx.put_metadata(key.as_bytes(), &value).map_err(Error::Transaction)?;
        }
        AlterIndexAction::ResetOptions { options } => {
            // Remove options from the schema
            for opt_key in options {
                schema.with_options.retain(|(k, _)| k != opt_key);
            }
            // Update in storage
            let key = format!("schema:index:{}", schema.name);
            let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
                .map_err(|e| Error::Execution(format!("Failed to serialize index schema: {e}")))?;
            tx.put_metadata(key.as_bytes(), &value).map_err(Error::Transaction)?;
        }
    }

    Ok(0)
}

/// Execute a TRUNCATE TABLE statement.
fn execute_truncate_table<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &TruncateTableNode,
) -> Result<u64> {
    let mut total_deleted = 0u64;

    for table_name in &node.names {
        // Check if table exists
        let table_schema = SchemaManager::get_table(tx, table_name)
            .map_err(|e| Error::Execution(e.to_string()))?;

        if table_schema.is_none() {
            return Err(Error::Execution(format!("Table '{}' does not exist", table_name)));
        }

        // Delete all entities with this label
        let entities = tx.iter_entities(Some(table_name)).map_err(Error::Transaction)?;
        let entity_ids: Vec<_> = entities.iter().map(|e| e.id).collect();

        for entity_id in entity_ids {
            tx.delete_entity(entity_id).map_err(Error::Transaction)?;
            total_deleted += 1;
        }

        // Handle CASCADE option - delete dependent data
        if node.cascade {
            // In a full implementation, we would also delete:
            // - Rows in tables that have foreign keys referencing this table
            // - Data from partitions of this table
            // For now, we just delete the entities
        }

        // Handle RESTART IDENTITY option
        if node.restart_identity {
            // In a full implementation, we would reset any identity/serial columns
            // For now, this is a no-op since we use UUID-based entity IDs
        }
    }

    Ok(total_deleted)
}

/// Clean up property index entries for a BTree index.
fn cleanup_btree_index_entries<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    schema: &crate::schema::IndexSchema,
) -> Result<u64> {
    use manifoldb_core::index::{IndexId, PropertyIndexScan};

    // Only handle single-column indexes for now
    if schema.columns.len() != 1 {
        return Ok(0);
    }

    let column_name = &schema.columns[0].expr;
    let table_name = &schema.table;

    // Create the index ID for this label + property combination
    let index_id = IndexId::from_label_property(table_name, column_name);

    // Get the range for all entries in this index
    let (start, end) = PropertyIndexScan::full_index_range(index_id);

    // Delete all entries in the range
    let deleted = tx
        .delete_property_index_range(&start, &end)
        .map_err(|e| Error::Execution(e.to_string()))?;

    Ok(deleted as u64)
}

/// Execute a CREATE VIEW statement.
///
/// Views are stored as query definitions that can be referenced like tables.
/// When a view is queried, its definition is expanded inline.
fn execute_create_view<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateViewNode,
    sql: &str,
) -> Result<u64> {
    // Extract the SELECT query from the original SQL
    // The query comes after "AS " in "CREATE [OR REPLACE] VIEW name [(columns)] AS SELECT ..."
    let query_sql = extract_view_query_sql(sql)?;

    // Create view schema with the raw SQL
    let schema = crate::schema::ViewSchema::new(
        node.name.clone(),
        node.columns.iter().map(|c| c.name.clone()).collect(),
        query_sql,
    );

    // Store the schema
    let key = format!("schema:view:{}", node.name);
    let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
        .map_err(|e| Error::Execution(format!("Failed to serialize view schema: {e}")))?;

    tx.put_metadata(key.as_bytes(), &value).map_err(Error::Transaction)?;

    // Add to views list
    let mut views = SchemaManager::list_views(tx).unwrap_or_default();
    if !views.contains(&node.name) {
        views.push(node.name.clone());
        let list_value = bincode::serde::encode_to_vec(&views, bincode::config::standard())
            .map_err(|e| Error::Execution(format!("Failed to serialize views list: {e}")))?;
        tx.put_metadata(b"schema:views_list", &list_value).map_err(Error::Transaction)?;
    } else if !node.or_replace {
        return Err(Error::Execution(format!("View already exists: {}", node.name)));
    }

    Ok(0) // DDL doesn't return row counts
}

/// Extract the SELECT query SQL from a CREATE VIEW statement.
///
/// Handles various CREATE VIEW syntax forms:
/// - CREATE VIEW name AS SELECT ...
/// - CREATE OR REPLACE VIEW name AS SELECT ...
/// - CREATE VIEW name (col1, col2) AS SELECT ...
fn extract_view_query_sql(sql: &str) -> Result<String> {
    // Find " AS " (case insensitive) to locate the start of the query
    let sql_upper = sql.to_uppercase();
    if let Some(as_pos) = sql_upper.find(" AS ") {
        let query_start = as_pos + 4; // Skip " AS "
        let query_sql = sql[query_start..].trim().to_string();
        if query_sql.is_empty() {
            return Err(Error::Parse("CREATE VIEW requires a query after AS".to_string()));
        }
        Ok(query_sql)
    } else {
        Err(Error::Parse("CREATE VIEW syntax error: missing AS clause".to_string()))
    }
}

/// Execute a DROP VIEW statement.
fn execute_drop_view<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &DropViewNode,
) -> Result<u64> {
    for view_name in &node.names {
        SchemaManager::drop_view(tx, view_name, node.if_exists)
            .map_err(|e| Error::Execution(e.to_string()))?;
    }
    Ok(0)
}

/// Execute a CREATE MATERIALIZED VIEW statement.
///
/// Materialized views store query results persistently and must be refreshed
/// explicitly to update their data.
fn execute_create_materialized_view<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateMaterializedViewNode,
    sql: &str,
) -> Result<u64> {
    use crate::schema::SchemaManager;

    // Extract the SELECT query from the original SQL
    let query_sql = extract_materialized_view_query_sql(sql)?;

    // Use SchemaManager for consistent storage
    let columns = node.columns.iter().map(|c| c.name.clone()).collect();
    SchemaManager::create_materialized_view(
        tx,
        &node.name,
        columns,
        &query_sql,
        node.if_not_exists,
    )
    .map_err(|e| Error::Execution(format!("Failed to create materialized view: {e}")))?;

    Ok(0)
}

/// Extract the SELECT query SQL from a CREATE MATERIALIZED VIEW statement.
fn extract_materialized_view_query_sql(sql: &str) -> Result<String> {
    let sql_upper = sql.to_uppercase();
    if let Some(as_pos) = sql_upper.find(" AS ") {
        let query_start = as_pos + 4;
        let query_sql = sql[query_start..].trim().to_string();
        if query_sql.is_empty() {
            return Err(Error::Parse(
                "CREATE MATERIALIZED VIEW requires a query after AS".to_string(),
            ));
        }
        Ok(query_sql)
    } else {
        Err(Error::Parse("CREATE MATERIALIZED VIEW syntax error: missing AS clause".to_string()))
    }
}

/// Execute a DROP MATERIALIZED VIEW statement.
fn execute_drop_materialized_view<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &DropMaterializedViewNode,
) -> Result<u64> {
    use crate::schema::SchemaManager;

    for view_name in &node.names {
        SchemaManager::drop_materialized_view(tx, view_name, node.if_exists)
            .map_err(|e| Error::Execution(format!("Failed to drop materialized view: {e}")))?;
    }
    Ok(0)
}

/// Execute a REFRESH MATERIALIZED VIEW statement.
///
/// Re-executes the materialized view's defining query and stores the results.
fn execute_refresh_materialized_view<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &RefreshMaterializedViewNode,
) -> Result<u64> {
    use crate::schema::{MaterializedViewRows, SchemaManager};

    // Check if the materialized view exists and get its schema
    let schema = SchemaManager::get_materialized_view(tx, &node.name)
        .map_err(|e| Error::Execution(format!("Failed to get materialized view: {e}")))?
        .ok_or_else(|| {
            Error::Execution(format!("Materialized view does not exist: {}", node.name))
        })?;

    // Execute the stored query to get fresh results
    let result = execute_query(tx, schema.query(), &[])?;

    // Extract column names from the result schema
    let result_columns: Vec<String> =
        result.schema().columns().into_iter().map(|c| c.to_string()).collect();

    // Convert the result rows to storable format
    let rows: Vec<Vec<Value>> = result.iter().map(|row| row.values().to_vec()).collect();
    let row_count = rows.len() as u64;

    // Store the cached rows
    let cached_rows = MaterializedViewRows { rows };
    SchemaManager::store_materialized_view_rows(tx, &node.name, cached_rows)
        .map_err(|e| Error::Execution(format!("Failed to store materialized view rows: {e}")))?;

    // Update the metadata
    let last_refreshed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    SchemaManager::update_materialized_view_data(
        tx,
        &node.name,
        last_refreshed,
        row_count,
        result_columns,
    )
    .map_err(|e| Error::Execution(format!("Failed to update materialized view data: {e}")))?;

    Ok(row_count)
}

/// Execute a CREATE COLLECTION statement.
///
/// Creates a vector collection with the specified vector configurations.
fn execute_create_collection<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateCollectionNode,
) -> Result<u64> {
    use crate::collection::{
        CollectionManager, CollectionName, DistanceType, HnswParams, IndexConfig,
        InvertedIndexParams, VectorConfig, VectorType,
    };
    use manifoldb_vector::distance::DistanceMetric;

    // Convert AST vector definitions to collection config
    let mut vector_configs = Vec::new();

    for vec_def in &node.vectors {
        // Parse vector type
        let vector_type = match &vec_def.vector_type {
            manifoldb_query::ast::VectorTypeDef::Vector { dimension } => {
                VectorType::Dense { dimension: *dimension as usize }
            }
            manifoldb_query::ast::VectorTypeDef::SparseVector { max_dimension } => {
                VectorType::Sparse { max_dimension: max_dimension.unwrap_or(0) }
            }
            manifoldb_query::ast::VectorTypeDef::MultiVector { token_dim } => {
                VectorType::Multi { token_dim: *token_dim as usize }
            }
            manifoldb_query::ast::VectorTypeDef::BinaryVector { bits } => {
                VectorType::Binary { bits: *bits as usize }
            }
        };

        // Parse distance metric from WITH options
        let mut distance_metric = DistanceMetric::Cosine;
        let mut m: Option<usize> = None;
        let mut ef_construction: Option<usize> = None;

        for (key, value) in &vec_def.with_options {
            match key.as_str() {
                "distance" => {
                    distance_metric = match value.to_lowercase().as_str() {
                        "euclidean" | "l2" => DistanceMetric::Euclidean,
                        "cosine" => DistanceMetric::Cosine,
                        "dot" | "dot_product" | "inner_product" => DistanceMetric::DotProduct,
                        _ => DistanceMetric::Cosine,
                    };
                }
                "m" => {
                    m = value.parse().ok();
                }
                "ef_construction" => {
                    ef_construction = value.parse().ok();
                }
                _ => {}
            }
        }

        // Parse index method
        let index_config = match vec_def.using.as_ref().map(|u| u.to_lowercase()).as_deref() {
            Some("hnsw") => {
                let mut hnsw_params = HnswParams::default();
                if let Some(m_val) = m {
                    hnsw_params = HnswParams::new(m_val);
                }
                if let Some(ef) = ef_construction {
                    hnsw_params = hnsw_params.with_ef_construction(ef);
                }
                IndexConfig::hnsw(hnsw_params)
            }
            Some("inverted") => IndexConfig::inverted(InvertedIndexParams::default()),
            Some("flat") | None => IndexConfig::flat(),
            Some(_) => IndexConfig::flat(),
        };

        // Build the vector config
        let config = VectorConfig {
            vector_type,
            distance: DistanceType::Dense(distance_metric),
            index: index_config,
        };

        vector_configs.push((vec_def.name.name.clone(), config));
    }

    // Create collection name
    let collection_name = CollectionName::new(&node.name)
        .map_err(|e| Error::Execution(format!("invalid collection name: {e}")))?;

    // Check if collection already exists and if_not_exists is set
    if node.if_not_exists && CollectionManager::exists(tx, &collection_name).unwrap_or(false) {
        return Ok(0);
    }

    // Create the collection using CollectionManager
    CollectionManager::create(tx, &collection_name, vector_configs)
        .map_err(|e| Error::Execution(e.to_string()))?;

    Ok(0)
}

/// Execute a DROP COLLECTION statement.
fn execute_drop_collection<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &DropCollectionNode,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};

    for name in &node.names {
        let collection_name = CollectionName::new(name)
            .map_err(|e| Error::Execution(format!("invalid collection name: {e}")))?;

        CollectionManager::delete(tx, &collection_name, node.if_exists)
            .map_err(|e| Error::Execution(e.to_string()))?;
    }
    Ok(0)
}

/// Evaluate a logical expression to a value.
///
/// # NULL semantics
///
/// This function follows SQL NULL semantics:
/// - Missing entity properties return NULL (sparse property model)
/// - Missing parameters return NULL (consistent with SQL prepared statements)
/// - Unsupported expressions return NULL (graceful degradation)
fn evaluate_expr(expr: &LogicalExpr, entity: &Entity, ctx: &ExecutionContext) -> Value {
    match expr {
        LogicalExpr::Literal(lit) => literal_to_value(lit),

        LogicalExpr::Column { qualifier, name } => {
            if name == "_rowid" {
                Value::Int(entity.id.as_u64() as i64)
            } else {
                // Try qualified name first (e.g., "target.id"), then unqualified
                let prop_value = if let Some(q) = qualifier {
                    // Try "qualifier.name" format (e.g., "target.id")
                    let qualified_name = format!("{}.{}", q, name);
                    entity
                        .get_property(&qualified_name)
                        .cloned()
                        .or_else(|| entity.get_property(name).cloned())
                } else {
                    entity.get_property(name).cloned()
                };
                // Missing properties return NULL - this is intentional for sparse property model
                prop_value.unwrap_or(Value::Null)
            }
        }

        LogicalExpr::Parameter(idx) => {
            // Missing parameters return NULL - consistent with SQL NULL semantics
            ctx.get_parameter(*idx as u32).cloned().unwrap_or(Value::Null)
        }

        LogicalExpr::Alias { expr, .. } => evaluate_expr(expr, entity, ctx),

        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr(left, entity, ctx);
            let rval = evaluate_expr(right, entity, ctx);
            evaluate_binary_op(op, &lval, &rval)
        }

        LogicalExpr::Wildcard => Value::Null, // Wildcard returns null in value context

        _ => Value::Null,
    }
}

/// Evaluate a logical expression against an entity with transaction access for subqueries.
///
/// This version handles scalar subqueries and other expressions that require
/// database access to evaluate.
fn evaluate_expr_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    expr: &LogicalExpr,
    entity: &Entity,
    ctx: &ExecutionContext,
) -> Value {
    match expr {
        // Scalar subquery - execute and return the result
        LogicalExpr::Subquery(subquery) => evaluate_scalar_subquery_with_tx(tx, subquery, ctx),

        // Binary operations - recursively evaluate operands
        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr_tx(tx, left, entity, ctx);
            let rval = evaluate_expr_tx(tx, right, entity, ctx);
            evaluate_binary_op(op, &lval, &rval)
        }

        // Alias - unwrap and evaluate inner
        LogicalExpr::Alias { expr: inner, .. } => evaluate_expr_tx(tx, inner, entity, ctx),

        // For non-subquery expressions, delegate to the basic version
        _ => evaluate_expr(expr, entity, ctx),
    }
}

/// Evaluate a literal expression (for INSERT VALUES).
///
/// # Errors
///
/// Returns an error if a parameter reference cannot be resolved or if the expression
/// type is not supported in literal context.
fn evaluate_literal_expr(expr: &LogicalExpr, ctx: &ExecutionContext) -> Result<Value> {
    match expr {
        LogicalExpr::Literal(lit) => Ok(literal_to_value(lit)),
        LogicalExpr::Parameter(idx) => ctx
            .get_parameter(*idx as u32)
            .cloned()
            .ok_or_else(|| Error::Execution(format!("missing parameter at index {}", idx))),
        other => Err(Error::Execution(format!(
            "unsupported expression type in VALUES clause: {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

/// Convert an AST literal to a Value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Boolean(b) => Value::Bool(*b),
        Literal::Integer(n) => Value::Int(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Vector(v) => Value::Vector(v.clone()),
        Literal::MultiVector(v) => Value::MultiVector(v.clone()),
    }
}

/// Evaluate an EXISTS subquery, returning true if any rows are returned.
///
/// This version works without a transaction by using the operator tree.
/// For queries that need to access database tables, use `evaluate_subquery_exists_with_tx`.
fn evaluate_subquery_exists(subquery: &LogicalPlan, ctx: &ExecutionContext) -> bool {
    use manifoldb_query::exec::build_operator_tree;
    use manifoldb_query::plan::PhysicalPlanner;

    // Convert logical plan to physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(subquery);

    // Build and execute the operator tree
    let Ok(mut op) = build_operator_tree(&physical_plan) else {
        return false;
    };

    if op.open(ctx).is_err() {
        return false;
    }

    // For EXISTS, we only need to check if there's at least one row
    let has_row = op.next().ok().flatten().is_some();
    let _ = op.close();

    has_row
}

/// Evaluate an EXISTS subquery with transaction access for database table scans.
fn evaluate_subquery_exists_with_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    subquery: &LogicalPlan,
    ctx: &ExecutionContext,
) -> bool {
    // Execute the subquery using the main executor
    let entities = match execute_logical_plan(tx, subquery, ctx) {
        Ok(entities) => entities,
        Err(_) => return false,
    };

    // EXISTS is true if there's at least one entity
    !entities.is_empty()
}

/// Evaluate an IN subquery, returning true if the value matches any row.
fn evaluate_subquery_in(val: &Value, subquery: &LogicalPlan, ctx: &ExecutionContext) -> bool {
    use manifoldb_query::exec::build_operator_tree;
    use manifoldb_query::plan::PhysicalPlanner;

    // Convert logical plan to physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(subquery);

    // Build and execute the operator tree
    let Ok(mut op) = build_operator_tree(&physical_plan) else {
        return false;
    };

    if op.open(ctx).is_err() {
        return false;
    }

    // Check if the value matches any row in the subquery
    let mut found = false;
    while let Ok(Some(row)) = op.next() {
        // Get the first column value from the row
        if let Some(subquery_val) = row.get(0) {
            if values_equal(val, subquery_val) {
                found = true;
                break;
            }
        }
    }

    let _ = op.close();
    found
}

/// Evaluate an IN subquery with transaction access for database table scans.
fn evaluate_subquery_in_with_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    val: &Value,
    subquery: &LogicalPlan,
    ctx: &ExecutionContext,
) -> bool {
    use manifoldb_query::plan::PhysicalPlanner;

    // Build physical plan for proper execution
    let catalog = build_planner_catalog(tx).unwrap_or_default();
    let planner = PhysicalPlanner::new().with_catalog(catalog);
    let physical_plan = planner.plan(subquery);

    // Execute through the physical plan path
    let result = match execute_physical_plan(tx, &physical_plan, subquery, ctx) {
        Ok(result) => result,
        Err(_) => return false,
    };

    // Check if the value matches any row's first column
    for row in result.rows() {
        if let Some(row_val) = row.get(0) {
            if values_equal(val, row_val) {
                return true;
            }
        }
    }

    false
}

/// Evaluate a scalar subquery, returning the single value or NULL.
fn evaluate_scalar_subquery(subquery: &LogicalPlan, ctx: &ExecutionContext) -> Value {
    use manifoldb_query::exec::build_operator_tree;
    use manifoldb_query::plan::PhysicalPlanner;

    // Convert logical plan to physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(subquery);

    // Build and execute the operator tree
    let Ok(mut op) = build_operator_tree(&physical_plan) else {
        return Value::Null;
    };

    if op.open(ctx).is_err() {
        return Value::Null;
    }

    // Get the first row
    let result = if let Ok(Some(row)) = op.next() {
        // Get the first column value
        row.get(0).cloned().unwrap_or(Value::Null)
    } else {
        Value::Null
    };

    let _ = op.close();
    result
}

/// Evaluate a scalar subquery with transaction access for database table scans.
fn evaluate_scalar_subquery_with_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    subquery: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Value {
    use manifoldb_query::plan::PhysicalPlanner;

    // Build physical plan for proper execution (including aggregates)
    let catalog = build_planner_catalog(tx).unwrap_or_default();
    let planner = PhysicalPlanner::new().with_catalog(catalog);
    let physical_plan = planner.plan(subquery);

    // Execute through the physical plan path to handle aggregates properly
    let result = match execute_physical_plan(tx, &physical_plan, subquery, ctx) {
        Ok(result) => result,
        Err(_) => return Value::Null,
    };

    // Return the first column value of the first row
    if let Some(row) = result.rows().first() {
        if let Some(val) = row.get(0) {
            return val.clone();
        }
    }

    Value::Null
}

/// Evaluate a predicate expression to a boolean.
fn evaluate_predicate(expr: &LogicalExpr, entity: &Entity, ctx: &ExecutionContext) -> bool {
    match expr {
        LogicalExpr::Literal(Literal::Boolean(b)) => *b,

        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr(left, entity, ctx);
            let rval = evaluate_expr(right, entity, ctx);

            use manifoldb_query::ast::BinaryOp;
            match op {
                // Comparison operators - return false if either operand is NULL (SQL semantics)
                // In SQL, NULL compared to anything returns NULL, which is treated as false in WHERE
                BinaryOp::Eq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        values_equal(&lval, &rval)
                    }
                }
                BinaryOp::NotEq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        !values_equal(&lval, &rval)
                    }
                }
                BinaryOp::Lt => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        compare_values(&lval, &rval) == std::cmp::Ordering::Less
                    }
                }
                BinaryOp::LtEq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        matches!(
                            compare_values(&lval, &rval),
                            std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                        )
                    }
                }
                BinaryOp::Gt => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        compare_values(&lval, &rval) == std::cmp::Ordering::Greater
                    }
                }
                BinaryOp::GtEq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        matches!(
                            compare_values(&lval, &rval),
                            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                        )
                    }
                }
                BinaryOp::And => {
                    evaluate_predicate(left, entity, ctx) && evaluate_predicate(right, entity, ctx)
                }
                BinaryOp::Or => {
                    evaluate_predicate(left, entity, ctx) || evaluate_predicate(right, entity, ctx)
                }
                BinaryOp::Like => {
                    // Simple LIKE implementation
                    if let (Value::String(s), Value::String(pattern)) = (&lval, &rval) {
                        simple_like_match(s, pattern)
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }

        LogicalExpr::UnaryOp { op, operand } => {
            use manifoldb_query::ast::UnaryOp;
            match op {
                UnaryOp::Not => !evaluate_predicate(operand, entity, ctx),
                UnaryOp::IsNull => {
                    matches!(evaluate_expr(operand, entity, ctx), Value::Null)
                }
                UnaryOp::IsNotNull => !matches!(evaluate_expr(operand, entity, ctx), Value::Null),
                _ => false,
            }
        }

        LogicalExpr::InList { expr, list, negated } => {
            let val = evaluate_expr(expr, entity, ctx);
            let in_list = list.iter().any(|item| {
                let item_val = evaluate_expr(item, entity, ctx);
                values_equal(&val, &item_val)
            });
            if *negated {
                !in_list
            } else {
                in_list
            }
        }

        LogicalExpr::Between { expr, low, high, negated } => {
            let val = evaluate_expr(expr, entity, ctx);
            let low_val = evaluate_expr(low, entity, ctx);
            let high_val = evaluate_expr(high, entity, ctx);

            let in_range = compare_values(&val, &low_val) != std::cmp::Ordering::Less
                && compare_values(&val, &high_val) != std::cmp::Ordering::Greater;

            if *negated {
                !in_range
            } else {
                in_range
            }
        }

        // SQL EXISTS subquery
        LogicalExpr::Exists { subquery, negated } => {
            // Execute the subquery using the operator tree
            let result = evaluate_subquery_exists(subquery, ctx);
            if *negated {
                !result
            } else {
                result
            }
        }

        // SQL IN subquery
        LogicalExpr::InSubquery { expr, subquery, negated } => {
            let val = evaluate_expr(expr, entity, ctx);
            if matches!(val, Value::Null) {
                return false; // NULL IN (...) is unknown, treated as false in WHERE
            }
            let result = evaluate_subquery_in(&val, subquery, ctx);
            if *negated {
                !result
            } else {
                result
            }
        }

        // SQL scalar subquery
        LogicalExpr::Subquery(subquery) => {
            // Scalar subquery in predicate context is truthy if not null
            let val = evaluate_scalar_subquery(subquery, ctx);
            !matches!(val, Value::Null)
        }

        _ => true, // Default to true for unhandled expressions
    }
}

/// Evaluate a predicate expression to a boolean with transaction access for subqueries.
///
/// This version can execute SQL subqueries (EXISTS, IN, scalar) that need to access
/// database tables. Use this version when you have a transaction available.
fn evaluate_predicate_with_tx<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    expr: &LogicalExpr,
    entity: &Entity,
    ctx: &ExecutionContext,
) -> bool {
    match expr {
        LogicalExpr::Literal(Literal::Boolean(b)) => *b,

        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr_tx(tx, left, entity, ctx);
            let rval = evaluate_expr_tx(tx, right, entity, ctx);

            use manifoldb_query::ast::BinaryOp;
            match op {
                BinaryOp::Eq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        values_equal(&lval, &rval)
                    }
                }
                BinaryOp::NotEq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        !values_equal(&lval, &rval)
                    }
                }
                BinaryOp::Lt => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        compare_values(&lval, &rval) == std::cmp::Ordering::Less
                    }
                }
                BinaryOp::LtEq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        matches!(
                            compare_values(&lval, &rval),
                            std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                        )
                    }
                }
                BinaryOp::Gt => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        compare_values(&lval, &rval) == std::cmp::Ordering::Greater
                    }
                }
                BinaryOp::GtEq => {
                    if matches!(lval, Value::Null) || matches!(rval, Value::Null) {
                        false
                    } else {
                        matches!(
                            compare_values(&lval, &rval),
                            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                        )
                    }
                }
                BinaryOp::And => {
                    evaluate_predicate_with_tx(tx, left, entity, ctx)
                        && evaluate_predicate_with_tx(tx, right, entity, ctx)
                }
                BinaryOp::Or => {
                    evaluate_predicate_with_tx(tx, left, entity, ctx)
                        || evaluate_predicate_with_tx(tx, right, entity, ctx)
                }
                BinaryOp::Like => {
                    if let (Value::String(s), Value::String(pattern)) = (&lval, &rval) {
                        simple_like_match(s, pattern)
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }

        LogicalExpr::UnaryOp { op, operand } => {
            use manifoldb_query::ast::UnaryOp;
            match op {
                UnaryOp::Not => !evaluate_predicate_with_tx(tx, operand, entity, ctx),
                UnaryOp::IsNull => {
                    matches!(evaluate_expr_tx(tx, operand, entity, ctx), Value::Null)
                }
                UnaryOp::IsNotNull => {
                    !matches!(evaluate_expr_tx(tx, operand, entity, ctx), Value::Null)
                }
                _ => false,
            }
        }

        LogicalExpr::InList { expr, list, negated } => {
            let val = evaluate_expr_tx(tx, expr, entity, ctx);
            let in_list = list.iter().any(|item| {
                let item_val = evaluate_expr_tx(tx, item, entity, ctx);
                values_equal(&val, &item_val)
            });
            if *negated {
                !in_list
            } else {
                in_list
            }
        }

        LogicalExpr::Between { expr, low, high, negated } => {
            let val = evaluate_expr_tx(tx, expr, entity, ctx);
            let low_val = evaluate_expr_tx(tx, low, entity, ctx);
            let high_val = evaluate_expr_tx(tx, high, entity, ctx);

            let in_range = compare_values(&val, &low_val) != std::cmp::Ordering::Less
                && compare_values(&val, &high_val) != std::cmp::Ordering::Greater;

            if *negated {
                !in_range
            } else {
                in_range
            }
        }

        // SQL EXISTS subquery - use transaction to execute against database
        LogicalExpr::Exists { subquery, negated } => {
            let result = evaluate_subquery_exists_with_tx(tx, subquery, ctx);
            if *negated {
                !result
            } else {
                result
            }
        }

        // SQL IN subquery - use transaction to execute against database
        LogicalExpr::InSubquery { expr, subquery, negated } => {
            let val = evaluate_expr_tx(tx, expr, entity, ctx);
            if matches!(val, Value::Null) {
                return false;
            }
            let result = evaluate_subquery_in_with_tx(tx, &val, subquery, ctx);
            if *negated {
                !result
            } else {
                result
            }
        }

        // SQL scalar subquery - use transaction to execute against database
        LogicalExpr::Subquery(subquery) => {
            let val = evaluate_scalar_subquery_with_tx(tx, subquery, ctx);
            !matches!(val, Value::Null)
        }

        _ => true,
    }
}

/// Evaluate a binary operation on two values.
fn evaluate_binary_op(op: &manifoldb_query::ast::BinaryOp, lval: &Value, rval: &Value) -> Value {
    use manifoldb_query::ast::BinaryOp;

    match op {
        BinaryOp::Add => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            (Value::String(a), Value::String(b)) => Value::String(format!("{a}{b}")),
            _ => Value::Null,
        },
        BinaryOp::Sub => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Mul => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Div => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        // Vector distance operators
        BinaryOp::EuclideanDistance => match (lval, rval) {
            (Value::Vector(a), Value::Vector(b)) if a.len() == b.len() => {
                let dist: f32 =
                    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
                Value::Float(f64::from(dist))
            }
            _ => Value::Null,
        },
        BinaryOp::CosineDistance => match (lval, rval) {
            (Value::Vector(a), Value::Vector(b)) if a.len() == b.len() => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    Value::Float(f64::MAX)
                } else {
                    Value::Float(f64::from(1.0 - (dot / (norm_a * norm_b))))
                }
            }
            _ => Value::Null,
        },
        BinaryOp::InnerProduct => match (lval, rval) {
            (Value::Vector(a), Value::Vector(b)) if a.len() == b.len() => {
                // Negative inner product for distance ordering (higher inner product = more similar)
                let prod: f32 = -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
                Value::Float(f64::from(prod))
            }
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

/// Check if two values are equal.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) => (*a as f64 - b).abs() < f64::EPSILON,
        (Value::Float(a), Value::Int(b)) => (a - *b as f64).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bytes(a), Value::Bytes(b)) => a == b,
        (Value::Vector(a), Value::Vector(b)) => a == b,
        _ => false,
    }
}

/// Compare two values for ordering.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// SQL LIKE pattern matching.
///
/// Supports:
/// - `%` matches any sequence of characters (including empty)
/// - `_` matches exactly one character
fn simple_like_match(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();

    let s_len = s_chars.len();
    let p_len = p_chars.len();

    // dp[i][j] = true if s[0..i] matches pattern[0..j]
    let mut dp = vec![vec![false; p_len + 1]; s_len + 1];

    // Empty pattern matches empty string
    dp[0][0] = true;

    // Handle patterns starting with %
    for j in 1..=p_len {
        if p_chars[j - 1] == '%' {
            dp[0][j] = dp[0][j - 1];
        } else {
            break;
        }
    }

    for i in 1..=s_len {
        for j in 1..=p_len {
            let p_char = p_chars[j - 1];

            if p_char == '%' {
                // % matches zero or more characters
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if p_char == '_' {
                // _ matches exactly one character
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                // Regular character - must match exactly
                dp[i][j] = dp[i - 1][j - 1] && s_chars[i - 1] == p_char;
            }
        }
    }

    dp[s_len][p_len]
}

/// Extract column name from an expression.
fn expr_to_column_name(expr: &LogicalExpr) -> String {
    match expr {
        LogicalExpr::Column { name, .. } => name.clone(),
        LogicalExpr::Alias { alias, .. } => alias.clone(),
        LogicalExpr::Wildcard => "*".to_string(),
        LogicalExpr::AggregateFunction { func, .. } => format!("{func:?}"),
        LogicalExpr::Literal(lit) => format!("{lit:?}"),
        _ => "?".to_string(),
    }
}

/// Evaluate a predicate expression for constraint validation.
///
/// This is a public wrapper around the internal `evaluate_predicate` function
/// for use by the constraint enforcement module.
///
/// Returns `true` if the predicate evaluates to true, `false` otherwise.
pub fn evaluate_predicate_for_constraint(
    expr: &LogicalExpr,
    entity: &Entity,
    ctx: &ExecutionContext,
) -> bool {
    evaluate_predicate(expr, entity, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_to_value() {
        assert_eq!(literal_to_value(&Literal::Null), Value::Null);
        assert_eq!(literal_to_value(&Literal::Boolean(true)), Value::Bool(true));
        assert_eq!(literal_to_value(&Literal::Integer(42)), Value::Int(42));
        assert_eq!(literal_to_value(&Literal::Float(1.5)), Value::Float(1.5));
        assert_eq!(
            literal_to_value(&Literal::String("hello".to_string())),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn test_simple_like_match() {
        // % wildcard
        assert!(simple_like_match("hello", "hello"));
        assert!(simple_like_match("hello", "hel%"));
        assert!(simple_like_match("hello", "%llo"));
        assert!(simple_like_match("hello", "%ell%"));
        assert!(!simple_like_match("hello", "world"));
        assert!(!simple_like_match("hello", "hi%"));

        // _ wildcard (single character)
        assert!(simple_like_match("hello", "h_llo"));
        assert!(simple_like_match("hello", "_ello"));
        assert!(simple_like_match("hello", "hell_"));
        assert!(simple_like_match("hello", "_____"));
        assert!(simple_like_match("Bob", "B_b"));
        assert!(!simple_like_match("hello", "h_lo")); // _ is one char, not two
        assert!(!simple_like_match("hello", "______")); // too many

        // Mixed wildcards
        assert!(simple_like_match("hello", "h_%"));
        assert!(simple_like_match("hello", "%_o"));
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Int(42), &Value::Int(42)));
        assert!(values_equal(&Value::Float(1.5), &Value::Float(1.5)));
        assert!(values_equal(&Value::Int(42), &Value::Float(42.0)));
        assert!(!values_equal(&Value::Int(42), &Value::Int(43)));
        assert!(values_equal(&Value::Null, &Value::Null));
    }

    #[test]
    fn test_compare_values() {
        use std::cmp::Ordering;

        assert_eq!(compare_values(&Value::Int(1), &Value::Int(2)), Ordering::Less);
        assert_eq!(compare_values(&Value::Int(2), &Value::Int(1)), Ordering::Greater);
        assert_eq!(compare_values(&Value::Int(1), &Value::Int(1)), Ordering::Equal);
        assert_eq!(compare_values(&Value::Null, &Value::Int(1)), Ordering::Less);
    }
}
