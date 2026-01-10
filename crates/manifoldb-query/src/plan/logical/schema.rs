//! Schema computation for logical plan nodes.
//!
//! This module provides methods for computing the output schema of logical plan nodes.
//! Each node type defines how it transforms its input schema(s) into an output schema.

use super::ddl::CreateTableNode;
use super::expr::LogicalExpr;
use super::node::LogicalPlan;
use super::relational::{JoinType, ScanNode, SetOpType, ValuesNode};
use super::type_infer::TypeResult;
use super::types::{PlanType, Schema, TypeContext, TypedColumn};

/// A trait for nodes that can provide schema information.
pub trait SchemaProvider {
    /// Returns the output schema of this node.
    ///
    /// The schema describes the columns that this node produces.
    fn output_schema(&self, input_schemas: &[&Schema]) -> TypeResult<Schema>;
}

/// Catalog interface for schema lookups.
///
/// This trait allows the schema computation to look up table schemas
/// from an external catalog.
pub trait SchemaCatalog {
    /// Looks up the schema for a table by name.
    fn table_schema(&self, table_name: &str) -> Option<Schema>;

    /// Looks up the schema for a view by name.
    fn view_schema(&self, view_name: &str) -> Option<Schema> {
        // Default: views are like tables
        self.table_schema(view_name)
    }
}

/// An empty catalog that returns no schemas.
///
/// Useful for testing or when schema information is not available.
pub struct EmptyCatalog;

impl SchemaCatalog for EmptyCatalog {
    fn table_schema(&self, _table_name: &str) -> Option<Schema> {
        None
    }
}

impl LogicalPlan {
    /// Computes the output schema of this plan node.
    ///
    /// This method recursively computes schemas through the plan tree.
    /// For leaf nodes (like Scan), it uses the provided catalog to look up
    /// table schemas. For transformation nodes, it computes the output
    /// schema based on the input schema(s) and the node's semantics.
    ///
    /// # Arguments
    ///
    /// * `catalog` - A catalog for looking up table schemas
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb_query::plan::logical::{LogicalPlan, EmptyCatalog};
    ///
    /// let plan = LogicalPlan::scan("users")
    ///     .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));
    ///
    /// let schema = plan.output_schema(&EmptyCatalog)?;
    /// ```
    pub fn output_schema(&self, catalog: &dyn SchemaCatalog) -> TypeResult<Schema> {
        match self {
            // ========== Leaf Nodes ==========
            Self::Scan(node) => scan_schema(node, catalog),

            Self::Values(node) => values_schema(node),

            Self::Empty { columns } => {
                // Empty relation has unknown types for all columns
                let cols = columns
                    .iter()
                    .map(|name| TypedColumn::new(name.clone(), PlanType::Any))
                    .collect();
                Ok(Schema::new(cols))
            }

            // ========== Unary Nodes ==========
            Self::Filter { input, .. } => {
                // Filter preserves the schema
                input.output_schema(catalog)
            }

            Self::Project { node, input } => {
                let input_schema = input.output_schema(catalog)?;
                project_schema(&node.exprs, &input_schema)
            }

            Self::Aggregate { node, input } => {
                let input_schema = input.output_schema(catalog)?;
                aggregate_schema(&node.group_by, &node.aggregates, &input_schema)
            }

            Self::Sort { input, .. } => {
                // Sort preserves the schema
                input.output_schema(catalog)
            }

            Self::Limit { input, .. } => {
                // Limit preserves the schema
                input.output_schema(catalog)
            }

            Self::Distinct { input, .. } => {
                // Distinct preserves the schema
                input.output_schema(catalog)
            }

            Self::Window { node, input } => {
                let input_schema = input.output_schema(catalog)?;
                window_schema(&node.window_exprs, &input_schema)
            }

            Self::Alias { alias, input } => {
                let input_schema = input.output_schema(catalog)?;
                Ok(input_schema.with_qualifier(alias.clone()))
            }

            Self::Unwind { node, input } => {
                let input_schema = input.output_schema(catalog)?;
                // Unwind adds a new column for the unwound element
                let element_type = {
                    let ctx = TypeContext::with_schema(input_schema.clone());
                    let list_type = node.list_expr.infer_type(&ctx)?;
                    list_type.element_type().cloned().unwrap_or(PlanType::Any)
                };
                let cols: Vec<_> = input_schema
                    .columns()
                    .iter()
                    .cloned()
                    .chain(std::iter::once(TypedColumn::new(node.alias.clone(), element_type)))
                    .collect();
                Ok(Schema::new(cols))
            }

            // ========== Binary Nodes ==========
            Self::Join { node, left, right } => {
                let left_schema = left.output_schema(catalog)?;
                let right_schema = right.output_schema(catalog)?;
                join_schema(&left_schema, &right_schema, &node.join_type)
            }

            Self::SetOp { node, left, right } => {
                let left_schema = left.output_schema(catalog)?;
                let right_schema = right.output_schema(catalog)?;
                set_op_schema(&left_schema, &right_schema, &node.op_type)
            }

            // ========== N-ary Nodes ==========
            Self::Union { inputs, .. } => {
                if inputs.is_empty() {
                    return Ok(Schema::empty());
                }
                // Union uses the schema of the first input
                inputs[0].output_schema(catalog)
            }

            // ========== Recursive/Subquery Nodes ==========
            Self::CallSubquery { subquery, input, .. } => {
                let input_schema = input.output_schema(catalog)?;
                let subquery_schema = subquery.output_schema(catalog)?;
                // Result is the combination of input and subquery output
                Ok(input_schema.merge(&subquery_schema))
            }

            Self::RecursiveCTE { node, initial, .. } => {
                // CTE schema comes from the initial query or explicit columns
                if node.columns.is_empty() {
                    initial.output_schema(catalog)
                } else {
                    let initial_schema = initial.output_schema(catalog)?;
                    // Map explicit column names to initial query types
                    let cols: Vec<_> = node
                        .columns
                        .iter()
                        .enumerate()
                        .map(|(i, name)| {
                            let data_type = initial_schema
                                .field_at(i)
                                .map(|c| c.data_type.clone())
                                .unwrap_or(PlanType::Any);
                            TypedColumn::new(name.clone(), data_type)
                        })
                        .collect();
                    Ok(Schema::new(cols))
                }
            }

            // ========== Graph Nodes ==========
            Self::Expand { node, input } => {
                let schema = input.output_schema(catalog)?;
                // Expand adds destination node and optionally edge variable
                let mut cols = schema.columns().to_vec();
                cols.push(TypedColumn::new(node.dst_var.clone(), PlanType::Node));
                if let Some(ref edge_var) = node.edge_var {
                    cols.push(TypedColumn::new(edge_var.clone(), PlanType::Edge));
                }
                Ok(Schema::new(cols))
            }

            Self::PathScan { node, input } => {
                let schema = input.output_schema(catalog)?;
                let mut cols = schema.columns().to_vec();
                // PathScan adds variables from each step (via the expand nodes)
                for step in &node.steps {
                    cols.push(TypedColumn::new(step.expand.dst_var.clone(), PlanType::Node));
                    if let Some(ref edge_var) = &step.expand.edge_var {
                        cols.push(TypedColumn::new(edge_var.clone(), PlanType::Edge));
                    }
                }
                // If tracking path, add a path variable
                if node.track_path {
                    cols.push(TypedColumn::new("_path", PlanType::Path));
                }
                Ok(Schema::new(cols))
            }

            Self::ShortestPath { node, input } => {
                let schema = input.output_schema(catalog)?;
                let mut cols = schema.columns().to_vec();
                // ShortestPath binds the path variable if specified
                if let Some(ref path_var) = node.path_variable {
                    cols.push(TypedColumn::new(path_var.clone(), PlanType::Path));
                }
                Ok(Schema::new(cols))
            }

            // ========== Vector Nodes ==========
            Self::AnnSearch { input, .. } => {
                let schema = input.output_schema(catalog)?;
                let mut cols = schema.columns().to_vec();
                // ANN search adds a distance column
                cols.push(TypedColumn::new("distance", PlanType::DoublePrecision));
                Ok(Schema::new(cols))
            }

            Self::VectorDistance { node, input } => {
                let schema = input.output_schema(catalog)?;
                let mut cols = schema.columns().to_vec();
                // Adds the computed distance column
                let alias = node.alias.clone().unwrap_or_else(|| "distance".to_string());
                cols.push(TypedColumn::new(alias, PlanType::DoublePrecision));
                Ok(Schema::new(cols))
            }

            Self::HybridSearch { input, .. } => {
                let schema = input.output_schema(catalog)?;
                let mut cols = schema.columns().to_vec();
                // Hybrid search adds a combined score column
                cols.push(TypedColumn::new("score", PlanType::DoublePrecision));
                Ok(Schema::new(cols))
            }

            // ========== DML Nodes ==========
            Self::Insert { returning, columns, .. } => {
                if returning.is_empty() {
                    // No returning clause - empty schema
                    Ok(Schema::empty())
                } else {
                    // Schema from RETURNING expressions
                    // We need to infer from a minimal context with column names
                    let ctx_schema = Schema::new(
                        columns
                            .iter()
                            .map(|c| TypedColumn::new(c.clone(), PlanType::Any))
                            .collect(),
                    );
                    let ctx = TypeContext::with_schema(ctx_schema);
                    let cols: Vec<_> = returning
                        .iter()
                        .map(|e| e.to_typed_column(&ctx))
                        .collect::<TypeResult<_>>()?;
                    Ok(Schema::new(cols))
                }
            }

            Self::Update { returning, assignments, .. } => {
                if returning.is_empty() {
                    Ok(Schema::empty())
                } else {
                    let ctx_schema = Schema::new(
                        assignments
                            .iter()
                            .map(|(c, _)| TypedColumn::new(c.clone(), PlanType::Any))
                            .collect(),
                    );
                    let ctx = TypeContext::with_schema(ctx_schema);
                    let cols: Vec<_> = returning
                        .iter()
                        .map(|e| e.to_typed_column(&ctx))
                        .collect::<TypeResult<_>>()?;
                    Ok(Schema::new(cols))
                }
            }

            Self::Delete { returning, .. } => {
                if returning.is_empty() {
                    Ok(Schema::empty())
                } else {
                    // Without table schema info, use Any types
                    let ctx = TypeContext::new();
                    let cols: Vec<_> = returning
                        .iter()
                        .map(|e| e.to_typed_column(&ctx))
                        .collect::<TypeResult<_>>()?;
                    Ok(Schema::new(cols))
                }
            }

            Self::MergeSql { .. } => {
                // MERGE doesn't return data (unless we add RETURNING support later)
                Ok(Schema::empty())
            }

            // ========== DDL Nodes ==========
            Self::CreateTable(node) => create_table_schema(node),

            Self::AlterTable(_)
            | Self::DropTable(_)
            | Self::TruncateTable(_)
            | Self::CreateIndex(_)
            | Self::AlterIndex(_)
            | Self::DropIndex(_)
            | Self::CreateCollection(_)
            | Self::DropCollection(_)
            | Self::CreateView(_)
            | Self::DropView(_)
            | Self::CreateMaterializedView(_)
            | Self::DropMaterializedView(_)
            | Self::RefreshMaterializedView(_)
            | Self::CreateSchema(_)
            | Self::AlterSchema(_)
            | Self::DropSchema(_)
            | Self::CreateFunction(_)
            | Self::DropFunction(_)
            | Self::CreateTrigger(_)
            | Self::DropTrigger(_) => {
                // DDL statements don't return data
                Ok(Schema::empty())
            }

            // ========== Graph DML Nodes ==========
            Self::GraphCreate { input, node } => {
                // Graph create can return created entities
                let base_schema = if let Some(inp) = input {
                    inp.output_schema(catalog)?
                } else {
                    Schema::empty()
                };
                // Add variables for created nodes/relationships
                let mut cols = base_schema.columns().to_vec();
                for create_node in &node.nodes {
                    if let Some(ref var) = create_node.variable {
                        cols.push(TypedColumn::new(var.clone(), PlanType::Node));
                    }
                }
                for create_rel in &node.relationships {
                    if let Some(ref var) = create_rel.rel_variable {
                        cols.push(TypedColumn::new(var.clone(), PlanType::Edge));
                    }
                }
                Ok(Schema::new(cols))
            }

            Self::GraphMerge { input, node } => {
                let base_schema = if let Some(inp) = input {
                    inp.output_schema(catalog)?
                } else {
                    Schema::empty()
                };
                // Add merged entity variable
                let mut cols = base_schema.columns().to_vec();
                let var_name = match &node.pattern {
                    super::graph::MergePatternSpec::Node { variable, .. } => variable.clone(),
                    super::graph::MergePatternSpec::Relationship { rel_variable, .. } => {
                        rel_variable.clone().unwrap_or_default()
                    }
                };
                let entity_type = match &node.pattern {
                    super::graph::MergePatternSpec::Node { .. } => PlanType::Node,
                    super::graph::MergePatternSpec::Relationship { .. } => PlanType::Edge,
                };
                if !var_name.is_empty() {
                    cols.push(TypedColumn::new(var_name, entity_type));
                }
                Ok(Schema::new(cols))
            }

            Self::GraphSet { input, .. }
            | Self::GraphDelete { input, .. }
            | Self::GraphRemove { input, .. }
            | Self::GraphForeach { input, .. } => {
                // These preserve the input schema
                input.output_schema(catalog)
            }

            // ========== Procedure Nodes ==========
            Self::ProcedureCall(node) => {
                // Schema from YIELD columns
                let cols: Vec<_> = node
                    .yield_columns
                    .iter()
                    .map(|yc| {
                        let name = yc.alias.clone().unwrap_or_else(|| yc.name.clone());
                        TypedColumn::new(name, PlanType::Any)
                    })
                    .collect();
                Ok(Schema::new(cols))
            }

            // ========== Transaction Nodes ==========
            Self::BeginTransaction(_)
            | Self::Commit(_)
            | Self::Rollback(_)
            | Self::Savepoint(_)
            | Self::ReleaseSavepoint(_)
            | Self::SetTransaction(_) => {
                // Transaction control doesn't return data
                Ok(Schema::empty())
            }

            // ========== Utility Nodes ==========
            Self::ExplainAnalyze(_) => {
                // EXPLAIN returns a single text column
                Ok(Schema::new(vec![TypedColumn::new("QUERY PLAN", PlanType::Text)]))
            }

            Self::Vacuum(_) | Self::Analyze(_) | Self::SetSession(_) | Self::Reset(_) => {
                Ok(Schema::empty())
            }

            Self::Copy(_) => {
                // COPY returns count of rows
                Ok(Schema::new(vec![TypedColumn::new("count", PlanType::BigInt)]))
            }

            Self::Show(_) => {
                // SHOW returns name and value columns
                Ok(Schema::new(vec![
                    TypedColumn::new("name", PlanType::Text),
                    TypedColumn::new("setting", PlanType::Text),
                ]))
            }

            Self::ShowProcedures(_) => {
                // SHOW PROCEDURES returns procedure metadata columns
                // Modeled after Neo4j's SHOW PROCEDURES output
                Ok(Schema::new(vec![
                    TypedColumn::new("name", PlanType::Text),
                    TypedColumn::new("description", PlanType::Text),
                    TypedColumn::new("mode", PlanType::Text),
                    TypedColumn::new("worksOnSystem", PlanType::Boolean),
                ]))
            }
        }
    }
}

/// Computes the schema for a scan node.
fn scan_schema(node: &ScanNode, catalog: &dyn SchemaCatalog) -> TypeResult<Schema> {
    // Try to get the schema from the catalog
    if let Some(schema) = catalog.table_schema(&node.table_name) {
        // Apply alias if present
        let schema = if let Some(ref alias) = node.alias {
            schema.with_qualifier(alias.clone())
        } else {
            schema.with_qualifier(node.table_name.clone())
        };

        // Apply column selection if present (projection)
        if let Some(ref projection) = node.projection {
            let selected: Option<Vec<_>> =
                projection.iter().map(|c| schema.field(c).cloned()).collect();
            if let Some(cols) = selected {
                return Ok(Schema::new(cols));
            }
        }

        Ok(schema)
    } else {
        // No catalog info - return schema with Any types for any specified columns
        // or an empty schema with the table name as qualifier
        if let Some(ref projection) = node.projection {
            let cols: Vec<_> = projection
                .iter()
                .map(|c| {
                    let col = TypedColumn::new(c.clone(), PlanType::Any);
                    if let Some(ref alias) = node.alias {
                        col.with_qualifier(alias.clone())
                    } else {
                        col.with_qualifier(node.table_name.clone())
                    }
                })
                .collect();
            Ok(Schema::new(cols))
        } else {
            // Unknown columns - return empty schema with a marker
            Ok(Schema::empty())
        }
    }
}

/// Computes the schema for a VALUES node.
fn values_schema(node: &ValuesNode) -> TypeResult<Schema> {
    if node.rows.is_empty() {
        return Ok(Schema::empty());
    }

    // Infer types from the first row
    let ctx = TypeContext::new();
    let cols: Vec<_> = node.rows[0]
        .iter()
        .enumerate()
        .map(|(i, expr)| {
            let data_type = expr.infer_type(&ctx).unwrap_or(PlanType::Any);
            TypedColumn::new(format!("column{}", i + 1), data_type)
        })
        .collect();

    Ok(Schema::new(cols))
}

/// Computes the schema for a projection.
fn project_schema(exprs: &[LogicalExpr], input_schema: &Schema) -> TypeResult<Schema> {
    let ctx = TypeContext::with_schema(input_schema.clone());

    let cols: Vec<_> = exprs
        .iter()
        .flat_map(|expr| match expr {
            LogicalExpr::Wildcard => {
                // Expand to all columns
                input_schema.columns().to_vec()
            }
            LogicalExpr::QualifiedWildcard(qualifier) => {
                // Expand to columns from the qualifier
                input_schema
                    .columns()
                    .iter()
                    .filter(|c| c.qualifier.as_deref() == Some(qualifier.as_str()))
                    .cloned()
                    .collect()
            }
            _ => {
                // Single expression
                vec![expr
                    .to_typed_column(&ctx)
                    .unwrap_or_else(|_| TypedColumn::new(expr.infer_name(), PlanType::Any))]
            }
        })
        .collect();

    Ok(Schema::new(cols))
}

/// Computes the schema for an aggregation.
fn aggregate_schema(
    group_by: &[LogicalExpr],
    aggregates: &[LogicalExpr],
    input_schema: &Schema,
) -> TypeResult<Schema> {
    let ctx = TypeContext::with_schema(input_schema.clone());

    let mut cols = Vec::new();

    // Group by columns come first
    for expr in group_by {
        cols.push(
            expr.to_typed_column(&ctx)
                .unwrap_or_else(|_| TypedColumn::new(expr.infer_name(), PlanType::Any)),
        );
    }

    // Then aggregate expressions
    for expr in aggregates {
        cols.push(
            expr.to_typed_column(&ctx)
                .unwrap_or_else(|_| TypedColumn::new(expr.infer_name(), PlanType::Any)),
        );
    }

    Ok(Schema::new(cols))
}

/// Computes the schema for window function additions.
fn window_schema(
    window_exprs: &[(LogicalExpr, String)],
    input_schema: &Schema,
) -> TypeResult<Schema> {
    let ctx = TypeContext::with_schema(input_schema.clone());

    // Window preserves input columns and adds window function results
    let mut cols = input_schema.columns().to_vec();

    for (expr, alias) in window_exprs {
        let data_type = expr.infer_type(&ctx).unwrap_or(PlanType::Any);
        cols.push(TypedColumn::new(alias.clone(), data_type));
    }

    Ok(Schema::new(cols))
}

/// Computes the schema for a join.
fn join_schema(left: &Schema, right: &Schema, join_type: &JoinType) -> TypeResult<Schema> {
    match join_type {
        // Semi/Anti joins only return left side columns
        JoinType::LeftSemi | JoinType::LeftAnti => Ok(left.clone()),
        // Right semi/anti return right side columns
        JoinType::RightSemi | JoinType::RightAnti => Ok(right.clone()),
        // Other joins combine both schemas
        _ => Ok(left.merge(right)),
    }
}

/// Computes the schema for a set operation.
fn set_op_schema(left: &Schema, _right: &Schema, _op_type: &SetOpType) -> TypeResult<Schema> {
    // Set operations use the schema of the left side
    // (both sides should have compatible schemas)
    Ok(left.clone())
}

/// Computes the schema for CREATE TABLE (returns the table schema).
fn create_table_schema(node: &CreateTableNode) -> TypeResult<Schema> {
    let cols: Vec<_> = node
        .columns
        .iter()
        .map(|col| {
            let data_type = PlanType::from(&col.data_type);
            let nullable = !col.constraints.iter().any(|c| {
                matches!(
                    c,
                    crate::ast::ColumnConstraint::NotNull
                        | crate::ast::ColumnConstraint::PrimaryKey
                )
            });
            TypedColumn::new(col.name.name.clone(), data_type).with_nullable(nullable)
        })
        .collect();

    Ok(Schema::new(cols))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::expr::LogicalExpr;

    struct TestCatalog {
        schemas: std::collections::HashMap<String, Schema>,
    }

    impl TestCatalog {
        fn new() -> Self {
            let mut schemas = std::collections::HashMap::new();
            schemas.insert(
                "users".to_string(),
                Schema::new(vec![
                    TypedColumn::new("id", PlanType::BigInt),
                    TypedColumn::new("name", PlanType::Text),
                    TypedColumn::new("age", PlanType::Integer),
                ]),
            );
            schemas.insert(
                "orders".to_string(),
                Schema::new(vec![
                    TypedColumn::new("id", PlanType::BigInt),
                    TypedColumn::new("user_id", PlanType::BigInt),
                    TypedColumn::new("amount", PlanType::DoublePrecision),
                ]),
            );
            Self { schemas }
        }
    }

    impl SchemaCatalog for TestCatalog {
        fn table_schema(&self, table_name: &str) -> Option<Schema> {
            self.schemas.get(table_name).cloned()
        }
    }

    #[test]
    fn test_scan_schema() {
        let catalog = TestCatalog::new();
        let plan = LogicalPlan::scan("users");

        let schema = plan.output_schema(&catalog).unwrap();
        assert_eq!(schema.len(), 3);
        assert!(schema.field("id").is_some());
        assert!(schema.field("name").is_some());
        assert!(schema.field("age").is_some());
    }

    #[test]
    fn test_scan_with_alias() {
        let catalog = TestCatalog::new();
        let plan = LogicalPlan::scan_aliased("users", "u");

        let schema = plan.output_schema(&catalog).unwrap();
        assert_eq!(schema.len(), 3);
        // Columns should be qualified with the alias
        let col = schema.field("id").unwrap();
        assert_eq!(col.qualifier, Some("u".to_string()));
    }

    #[test]
    fn test_filter_preserves_schema() {
        let catalog = TestCatalog::new();
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let schema = plan.output_schema(&catalog).unwrap();
        assert_eq!(schema.len(), 3);
    }

    #[test]
    fn test_project_schema() {
        let catalog = TestCatalog::new();
        let plan = LogicalPlan::scan("users")
            .project(vec![LogicalExpr::column("id"), LogicalExpr::column("name")]);

        let schema = plan.output_schema(&catalog).unwrap();
        assert_eq!(schema.len(), 2);
    }

    #[test]
    fn test_aggregate_schema() {
        let catalog = TestCatalog::new();
        let plan = LogicalPlan::scan("users").aggregate(
            vec![LogicalExpr::column("name")],
            vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
        );

        let schema = plan.output_schema(&catalog).unwrap();
        assert_eq!(schema.len(), 2); // name + count
    }

    #[test]
    fn test_join_schema() {
        let catalog = TestCatalog::new();
        let plan = LogicalPlan::scan("users").inner_join(
            LogicalPlan::scan("orders"),
            LogicalExpr::qualified_column("users", "id")
                .eq(LogicalExpr::qualified_column("orders", "user_id")),
        );

        let schema = plan.output_schema(&catalog).unwrap();
        // Should have columns from both tables
        assert_eq!(schema.len(), 6); // 3 from users + 3 from orders
    }

    #[test]
    fn test_empty_catalog() {
        let catalog = EmptyCatalog;
        let plan = LogicalPlan::scan("unknown_table");

        let schema = plan.output_schema(&catalog).unwrap();
        // With no catalog info, returns empty schema
        assert!(schema.is_empty());
    }
}
