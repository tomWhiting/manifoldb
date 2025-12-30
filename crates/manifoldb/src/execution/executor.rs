//! SQL query and statement execution.
//!
//! This module provides functions to execute SQL queries and statements
//! against the storage layer.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{Entity, Value};
use manifoldb_query::ast::DistanceMetric;
use manifoldb_query::ast::Literal;
use manifoldb_query::exec::operators::{
    BruteForceSearchOp, HashAggregateOp, HashJoinOp, NestedLoopJoinOp, SetOpOp, UnionOp, ValuesOp,
};
use manifoldb_query::exec::row::{Row, Schema};
use manifoldb_query::exec::{ExecutionContext, Operator, ResultSet};
use manifoldb_query::plan::logical::{AnnSearchNode, ExpandNode, PathScanNode, VectorDistanceNode};
use manifoldb_query::plan::logical::{
    CreateCollectionNode, CreateIndexNode, CreateTableNode, DropCollectionNode, DropIndexNode,
    DropTableNode, JoinType, LogicalExpr, SetOpNode, UnionNode,
};
use manifoldb_query::plan::physical::{IndexInfo, IndexType, PlannerCatalog};
use manifoldb_query::plan::{LogicalPlan, PhysicalPlan, PhysicalPlanner, PlanBuilder};
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
    // Parse SQL using ExtendedParser to support MATCH syntax
    let stmt = ExtendedParser::parse_single(sql)?;

    // Build logical plan
    let mut builder = PlanBuilder::new();
    let logical_plan = builder.build_statement(&stmt).map_err(|e| Error::Parse(e.to_string()))?;

    // Build catalog with available indexes from schema
    let catalog = build_planner_catalog(tx)?;

    // Build physical plan with catalog for index selection
    let planner = PhysicalPlanner::new().with_catalog(catalog);
    let physical_plan = planner.plan(&logical_plan);

    // Create execution context with parameters and row limit
    let ctx = create_context_with_limit(params, max_rows_in_memory);

    // Execute the plan against storage
    execute_physical_plan(tx, &physical_plan, &logical_plan, &ctx)
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
        LogicalPlan::Insert { table, columns, input, .. } => {
            execute_insert(tx, table, columns, input, &ctx)
        }
        LogicalPlan::Update { table, assignments, filter, .. } => {
            execute_update(tx, table, assignments, filter, &ctx)
        }
        LogicalPlan::Delete { table, filter, .. } => execute_delete(tx, table, filter, &ctx),

        // DDL statements
        LogicalPlan::CreateTable(node) => execute_create_table(tx, node),
        LogicalPlan::DropTable(node) => execute_drop_table(tx, node),
        LogicalPlan::CreateIndex(node) => execute_create_index(tx, node),
        LogicalPlan::DropIndex(node) => execute_drop_index(tx, node),
        LogicalPlan::CreateCollection(node) => execute_create_collection(tx, node),
        LogicalPlan::DropCollection(node) => execute_drop_collection(tx, node),

        _ => {
            // For SELECT, we shouldn't be here but handle gracefully
            Err(Error::Execution("Expected DML or DDL statement".to_string()))
        }
    }
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
        LogicalPlan::Insert { table, columns, input, .. } => {
            execute_insert(tx, table, columns, input, &ctx)
        }
        LogicalPlan::Update { table, assignments, filter, .. } => {
            execute_update(tx, table, assignments, filter, &ctx)
        }
        LogicalPlan::Delete { table, filter, .. } => execute_delete(tx, table, filter, &ctx),

        // DDL statements
        LogicalPlan::CreateTable(node) => execute_create_table(tx, node),
        LogicalPlan::DropTable(node) => execute_drop_table(tx, node),
        LogicalPlan::CreateIndex(node) => execute_create_index(tx, node),
        LogicalPlan::DropIndex(node) => execute_drop_index(tx, node),
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
                let columns: Vec<String> =
                    schema.columns.iter().map(|c| c.expr.clone()).collect();

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
                let filtered_rows: Vec<Row> = result
                    .into_rows()
                    .into_iter()
                    .filter(|row| {
                        let val = evaluate_row_expr(&node.predicate, row);
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
                            .map(|expr| evaluate_expr_on_row(expr, row, &result_schema, ctx))
                            .collect();
                        Row::new(Arc::clone(&new_schema), values)
                    })
                    .collect();

                return Ok(Some(ResultSet::with_rows(new_schema, rows)));
            }
            Ok(None)
        }

        _ => {
            // No index scan in this plan - fall back to logical execution
            Ok(None)
        }
    }
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
                            .map(|expr| evaluate_expr_on_row(expr, row, &result_schema, ctx))
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
                    let values: Vec<Value> =
                        node.exprs.iter().map(|expr| evaluate_expr(expr, entity, ctx)).collect();
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
            // Execute a table scan and convert to result set for graph traversal input
            let label = &scan_node.table_name;
            let alias = scan_node.alias.as_deref().unwrap_or(label);
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
            let entities = tx.iter_entities(Some(label)).map_err(Error::Transaction)?;
            Ok(entities)
        }

        LogicalPlan::Filter { node, input } => {
            let entities = execute_logical_plan(tx, input, ctx)?;

            // Filter entities based on predicate
            let filtered: Vec<Entity> = entities
                .into_iter()
                .filter(|entity| evaluate_predicate(&node.predicate, entity, ctx))
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
        | LogicalPlan::DropTable(_)
        | LogicalPlan::CreateIndex(_)
        | LogicalPlan::DropIndex(_)
        | LogicalPlan::CreateCollection(_)
        | LogicalPlan::DropCollection(_) => Err(Error::Execution(
            "DDL statements should be executed via execute_statement, not execute_logical_plan"
                .to_string(),
        )),

        LogicalPlan::HybridSearch { input, .. } => {
            // Hybrid search in entity context: execute the input, scoring handled by physical plan
            execute_logical_plan(tx, input, ctx)
        }
    }
}

/// Execute an INSERT statement.
fn execute_insert<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    columns: &[String],
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};
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
            // Create a new entity with the table name as label
            let mut entity = tx.create_entity().map_err(Error::Transaction)?;
            entity = entity.with_label(table);

            // Collect vectors to store separately (for collections)
            let mut vectors_to_store: Vec<(String, VectorData)> = Vec::new();

            // Set properties from columns and values
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

                    entity = entity.with_property(col, value);
                }
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
fn execute_update<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    assignments: &[(String, LogicalExpr)],
    filter: &Option<LogicalExpr>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};

    // Check if this is a collection with named vectors
    let collection = CollectionName::new(table)
        .ok()
        .and_then(|name| CollectionManager::get(tx, &name).ok().flatten());

    // Get all entities with this label
    let entities = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

    let mut count = 0;

    for entity in entities {
        // Check if entity matches filter
        let matches = match filter {
            Some(pred) => evaluate_predicate(pred, &entity, ctx),
            None => true,
        };

        if matches {
            // Clone the old entity before modifying
            let old_entity = entity.clone();
            let mut updated_entity = entity;

            // Collect vectors to update separately (for collections)
            let mut vectors_to_update: Vec<(String, manifoldb_vector::types::VectorData)> =
                Vec::new();

            // Apply assignments
            for (col, expr) in assignments {
                let value = evaluate_expr(expr, &updated_entity, ctx);

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

                updated_entity.set_property(col, value);
            }

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
    }

    Ok(count)
}

/// Execute a DELETE statement.
fn execute_delete<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    filter: &Option<LogicalExpr>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    use crate::collection::{CollectionManager, CollectionName};

    // Check if this is a collection with named vectors
    let collection = CollectionName::new(table)
        .ok()
        .and_then(|name| CollectionManager::get(tx, &name).ok().flatten());

    // Get all entities with this label
    let entities = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

    let mut count = 0;

    for entity in entities {
        // Check if entity matches filter
        let matches = match filter {
            Some(pred) => evaluate_predicate(pred, &entity, ctx),
            None => true,
        };

        if matches {
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
    }

    Ok(count)
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
            let entities = tx.iter_entities(Some(label)).map_err(Error::Transaction)?;
            Ok((entities, alias.to_string()))
        }
        LogicalPlan::Alias { alias, input } => {
            let (entities, _) = execute_join_input(tx, input, ctx)?;
            Ok((entities, alias.clone()))
        }
        LogicalPlan::Filter { node, input } => {
            let (entities, alias) = execute_join_input(tx, input, ctx)?;
            let filtered: Vec<Entity> = entities
                .into_iter()
                .filter(|entity| evaluate_predicate(&node.predicate, entity, ctx))
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

/// Execute a CREATE TABLE statement.
fn execute_create_table<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    node: &CreateTableNode,
) -> Result<u64> {
    SchemaManager::create_table(tx, node).map_err(|e| Error::Execution(e.to_string()))?;
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

        LogicalExpr::Column { name, .. } => {
            if name == "_rowid" {
                Value::Int(entity.id.as_u64() as i64)
            } else {
                // Missing properties return NULL - this is intentional for sparse property model
                entity.get_property(name).cloned().unwrap_or(Value::Null)
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

        _ => true, // Default to true for unhandled expressions
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
