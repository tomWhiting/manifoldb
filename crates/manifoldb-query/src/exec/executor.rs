//! Main query executor.
//!
//! This module provides the [`Executor`] that builds and runs
//! operator trees from physical plans.

use std::sync::Arc;

use crate::plan::physical::PhysicalPlan;

use super::context::ExecutionContext;
use super::operator::{BoxedOperator, OperatorResult, OperatorState};
use super::operators::{
    aggregate::{HashAggregateOp, SortMergeAggregateOp},
    filter::FilterOp,
    graph::{GraphExpandOp, GraphPathScanOp},
    join::{HashJoinOp, MergeJoinOp, NestedLoopJoinOp},
    limit::LimitOp,
    project::ProjectOp,
    scan::{FullScanOp, IndexRangeScanOp, IndexScanOp},
    set_ops::{SetOpOp, UnionOp},
    sort::SortOp,
    values::{EmptyOp, ValuesOp},
    vector::{BruteForceSearchOp, HnswSearchOp},
};
use super::result::{QueryResult, ResultSet, ResultSetBuilder};
use super::row::{Row, Schema};

/// The main query executor.
///
/// Builds an operator tree from a physical plan and executes it
/// to produce results.
pub struct Executor {
    /// The root operator of the tree.
    root: BoxedOperator,
    /// Execution context.
    ctx: ExecutionContext,
    /// Whether the executor has been opened.
    opened: bool,
}

impl Executor {
    /// Creates a new executor for the given physical plan.
    pub fn new(plan: &PhysicalPlan, ctx: ExecutionContext) -> OperatorResult<Self> {
        let root = build_operator_tree(plan)?;
        Ok(Self { root, ctx, opened: false })
    }

    /// Returns the output schema.
    #[must_use]
    pub fn schema(&self) -> Arc<Schema> {
        self.root.schema()
    }

    /// Opens the executor and prepares it to produce rows.
    pub fn open(&mut self) -> OperatorResult<()> {
        if !self.opened {
            self.root.open(&self.ctx)?;
            self.opened = true;
        }
        Ok(())
    }

    /// Returns the next row, or `None` if there are no more rows.
    pub fn next(&mut self) -> OperatorResult<Option<Row>> {
        if !self.opened {
            self.open()?;
        }

        // Check for cancellation
        if self.ctx.is_cancelled() {
            return Ok(None);
        }

        let row = self.root.next()?;

        // Update stats
        if row.is_some() {
            self.ctx.record_rows_produced(1);
        }

        Ok(row)
    }

    /// Closes the executor and releases resources.
    pub fn close(&mut self) -> OperatorResult<()> {
        if self.opened {
            self.root.close()?;
            self.opened = false;
        }
        Ok(())
    }

    /// Returns the execution context.
    #[must_use]
    pub fn context(&self) -> &ExecutionContext {
        &self.ctx
    }

    /// Returns the current state.
    #[must_use]
    pub fn state(&self) -> OperatorState {
        self.root.state()
    }

    /// Executes the query and collects all results.
    pub fn execute(&mut self) -> OperatorResult<QueryResult> {
        self.open()?;

        let schema = self.root.schema();
        let mut builder = ResultSetBuilder::new(schema);

        while let Some(row) = self.next()? {
            builder.push(row);
        }

        self.close()?;

        Ok(QueryResult::select(builder.build()))
    }

    /// Executes and returns just the rows as a vector.
    pub fn collect(&mut self) -> OperatorResult<Vec<Row>> {
        self.open()?;

        let mut rows = Vec::new();
        while let Some(row) = self.next()? {
            rows.push(row);
        }

        self.close()?;
        Ok(rows)
    }

    /// Counts the number of result rows without materializing them.
    pub fn count(&mut self) -> OperatorResult<usize> {
        self.open()?;

        let mut count = 0;
        while self.next()?.is_some() {
            count += 1;
        }

        self.close()?;
        Ok(count)
    }

    /// Returns the first row if any.
    pub fn first(&mut self) -> OperatorResult<Option<Row>> {
        self.open()?;
        let row = self.next()?;
        self.close()?;
        Ok(row)
    }
}

/// Builds an operator tree from a physical plan.
fn build_operator_tree(plan: &PhysicalPlan) -> OperatorResult<BoxedOperator> {
    match plan {
        // Scan operations
        PhysicalPlan::FullScan(node) => Ok(Box::new(FullScanOp::new((**node).clone()))),

        PhysicalPlan::IndexScan(node) => Ok(Box::new(IndexScanOp::new((**node).clone()))),

        PhysicalPlan::IndexRangeScan(node) => Ok(Box::new(IndexRangeScanOp::new((**node).clone()))),

        PhysicalPlan::Values { rows, .. } => {
            // Convert LogicalExpr rows to Value rows
            // For now, use empty schema - actual evaluation would happen here
            let schema = Arc::new(Schema::new(
                (0..rows.first().map_or(0, |r| r.len())).map(|i| format!("col_{i}")).collect(),
            ));
            Ok(Box::new(ValuesOp::new(schema, Vec::new())))
        }

        PhysicalPlan::Empty { columns } => Ok(Box::new(EmptyOp::with_columns(columns.clone()))),

        // Unary operators
        PhysicalPlan::Filter { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(FilterOp::new(node.predicate.clone(), input_op)))
        }

        PhysicalPlan::Project { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(ProjectOp::new(node.exprs.clone(), input_op)))
        }

        PhysicalPlan::Sort { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(SortOp::new(node.order_by.clone(), input_op)))
        }

        PhysicalPlan::Limit { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(LimitOp::new(node.limit, node.offset, input_op)))
        }

        PhysicalPlan::HashDistinct { on_columns, input, .. } => {
            // Implement as a hash aggregate with just grouping columns
            let input_op = build_operator_tree(input)?;
            let group_by = on_columns.clone().unwrap_or_default();
            Ok(Box::new(HashAggregateOp::new(group_by, vec![], None, input_op)))
        }

        PhysicalPlan::HashAggregate { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(HashAggregateOp::new(
                node.group_by.clone(),
                node.aggregates.clone(),
                node.having.clone(),
                input_op,
            )))
        }

        PhysicalPlan::SortMergeAggregate { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(SortMergeAggregateOp::new(
                node.group_by.clone(),
                node.aggregates.clone(),
                node.having.clone(),
                input_op,
            )))
        }

        // Join operators
        PhysicalPlan::NestedLoopJoin { node, left, right } => {
            let left_op = build_operator_tree(left)?;
            let right_op = build_operator_tree(right)?;
            Ok(Box::new(NestedLoopJoinOp::new(
                node.join_type,
                node.condition.clone(),
                left_op,
                right_op,
            )))
        }

        PhysicalPlan::HashJoin { node, build, probe } => {
            let build_op = build_operator_tree(build)?;
            let probe_op = build_operator_tree(probe)?;
            Ok(Box::new(HashJoinOp::new(
                node.join_type,
                node.build_keys.clone(),
                node.probe_keys.clone(),
                node.filter.clone(),
                build_op,
                probe_op,
            )))
        }

        PhysicalPlan::MergeJoin { node, left, right } => {
            let left_op = build_operator_tree(left)?;
            let right_op = build_operator_tree(right)?;
            Ok(Box::new(MergeJoinOp::new(
                node.join_type,
                node.left_keys.clone(),
                node.right_keys.clone(),
                left_op,
                right_op,
            )))
        }

        // Set operations
        PhysicalPlan::SetOp { op_type, left, right, .. } => {
            let left_op = build_operator_tree(left)?;
            let right_op = build_operator_tree(right)?;
            Ok(Box::new(SetOpOp::new(*op_type, left_op, right_op)))
        }

        PhysicalPlan::Union { all, inputs, .. } => {
            if inputs.is_empty() {
                let schema = Arc::new(Schema::empty());
                return Ok(Box::new(EmptyOp::new(schema)));
            }

            let input_ops: Vec<BoxedOperator> =
                inputs.iter().map(build_operator_tree).collect::<Result<_, _>>()?;
            Ok(Box::new(UnionOp::new(input_ops, *all)))
        }

        // Vector operations
        PhysicalPlan::HnswSearch { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(HnswSearchOp::new(
                node.vector_column.clone(),
                node.query_vector.clone(),
                node.metric,
                node.k,
                node.ef_search,
                node.include_distance,
                node.distance_alias.clone(),
                input_op,
            )))
        }

        PhysicalPlan::BruteForceSearch { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(BruteForceSearchOp::new(
                node.vector_column.clone(),
                node.query_vector.clone(),
                node.metric,
                node.k,
                node.include_distance,
                node.distance_alias.clone(),
                input_op,
            )))
        }

        // Graph operations
        PhysicalPlan::GraphExpand { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(GraphExpandOp::new((**node).clone(), input_op)))
        }

        PhysicalPlan::GraphPathScan { node, input } => {
            let input_op = build_operator_tree(input)?;
            Ok(Box::new(GraphPathScanOp::new(
                node.steps.clone(),
                node.all_paths,
                node.track_path,
                input_op,
            )))
        }

        // DML operations (not fully implemented)
        PhysicalPlan::Insert { columns, .. } => {
            Ok(Box::new(EmptyOp::with_columns(columns.clone())))
        }

        PhysicalPlan::Update { .. } => Ok(Box::new(EmptyOp::with_columns(vec![]))),

        PhysicalPlan::Delete { .. } => Ok(Box::new(EmptyOp::with_columns(vec![]))),

        // DDL operations are handled at a higher level, not as operators
        PhysicalPlan::CreateTable(_)
        | PhysicalPlan::DropTable(_)
        | PhysicalPlan::CreateIndex(_)
        | PhysicalPlan::DropIndex(_)
        | PhysicalPlan::CreateCollection(_)
        | PhysicalPlan::DropCollection(_) => Ok(Box::new(EmptyOp::with_columns(vec![]))),
    }
}

/// Convenience function to execute a plan and get results.
///
/// Creates a default execution context and runs the plan to completion.
pub fn execute_plan(plan: &PhysicalPlan) -> OperatorResult<ResultSet> {
    let ctx = ExecutionContext::new();
    let mut executor = Executor::new(plan, ctx)?;
    let result = executor.execute()?;
    result
        .into_select()
        .ok_or_else(|| crate::error::ParseError::Unsupported("Expected SELECT result".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::LogicalExpr;
    use crate::plan::logical::SortOrder;
    use crate::plan::physical::{
        FilterExecNode, FullScanNode, LimitExecNode, ProjectExecNode, SortExecNode,
    };

    fn make_scan_plan() -> PhysicalPlan {
        PhysicalPlan::FullScan(Box::new(
            FullScanNode::new("users").with_projection(vec!["id".to_string(), "name".to_string()]),
        ))
    }

    #[test]
    fn executor_empty() {
        let plan = PhysicalPlan::Empty { columns: vec!["a".to_string()] };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();

        assert_eq!(executor.count().unwrap(), 0);
    }

    #[test]
    fn executor_filter() {
        let scan = make_scan_plan();
        let plan = PhysicalPlan::Filter {
            node: FilterExecNode::new(LogicalExpr::column("id").gt(LogicalExpr::integer(5))),
            input: Box::new(scan),
        };

        let ctx = ExecutionContext::new();
        let executor = Executor::new(&plan, ctx).unwrap();

        // Just verify it builds
        assert_eq!(executor.schema().columns(), &["id", "name"]);
    }

    #[test]
    fn executor_project() {
        let scan = make_scan_plan();
        let plan = PhysicalPlan::Project {
            node: ProjectExecNode::new(vec![LogicalExpr::column("name")]),
            input: Box::new(scan),
        };

        let ctx = ExecutionContext::new();
        let executor = Executor::new(&plan, ctx).unwrap();

        assert_eq!(executor.schema().columns(), &["name"]);
    }

    #[test]
    fn executor_limit() {
        let scan = make_scan_plan();
        let plan = PhysicalPlan::Limit { node: LimitExecNode::limit(10), input: Box::new(scan) };

        let ctx = ExecutionContext::new();
        let executor = Executor::new(&plan, ctx).unwrap();

        assert_eq!(executor.schema().columns(), &["id", "name"]);
    }

    #[test]
    fn executor_sort() {
        let scan = make_scan_plan();
        let plan = PhysicalPlan::Sort {
            node: SortExecNode::new(vec![SortOrder::asc(LogicalExpr::column("name"))]),
            input: Box::new(scan),
        };

        let ctx = ExecutionContext::new();
        let executor = Executor::new(&plan, ctx).unwrap();

        assert_eq!(executor.schema().columns(), &["id", "name"]);
    }

    #[test]
    fn executor_cancellation() {
        let plan = make_scan_plan();
        let ctx = ExecutionContext::new();
        ctx.cancel();

        let mut executor = Executor::new(&plan, ctx).unwrap();
        executor.open().unwrap();

        // Should return None due to cancellation
        assert!(executor.next().unwrap().is_none());
    }

    #[test]
    fn executor_stats() {
        let plan = PhysicalPlan::Empty { columns: vec!["x".to_string()] };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();
        executor.execute().unwrap();

        assert!(executor.context().stats().elapsed().as_nanos() > 0);
    }

    #[test]
    fn executor_union() {
        use crate::plan::physical::Cost;

        // Union of two empty sources
        let plan = PhysicalPlan::Union {
            all: false,
            cost: Cost::default(),
            inputs: vec![
                PhysicalPlan::Empty { columns: vec!["x".to_string()] },
                PhysicalPlan::Empty { columns: vec!["x".to_string()] },
            ],
        };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();
        assert_eq!(executor.count().unwrap(), 0);
    }

    #[test]
    fn executor_union_all() {
        use crate::plan::physical::Cost;

        // Union ALL of empty sources
        let plan = PhysicalPlan::Union {
            all: true,
            cost: Cost::default(),
            inputs: vec![
                PhysicalPlan::Empty { columns: vec!["x".to_string()] },
                PhysicalPlan::Empty { columns: vec!["x".to_string()] },
            ],
        };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();
        assert_eq!(executor.count().unwrap(), 0);
    }

    #[test]
    fn executor_set_op_intersect() {
        use crate::plan::logical::SetOpType;
        use crate::plan::physical::Cost;

        let plan = PhysicalPlan::SetOp {
            op_type: SetOpType::Intersect,
            cost: Cost::default(),
            left: Box::new(PhysicalPlan::Empty { columns: vec!["x".to_string()] }),
            right: Box::new(PhysicalPlan::Empty { columns: vec!["x".to_string()] }),
        };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();
        assert_eq!(executor.count().unwrap(), 0);
    }

    #[test]
    fn executor_set_op_except() {
        use crate::plan::logical::SetOpType;
        use crate::plan::physical::Cost;

        let plan = PhysicalPlan::SetOp {
            op_type: SetOpType::Except,
            cost: Cost::default(),
            left: Box::new(PhysicalPlan::Empty { columns: vec!["x".to_string()] }),
            right: Box::new(PhysicalPlan::Empty { columns: vec!["x".to_string()] }),
        };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();
        assert_eq!(executor.count().unwrap(), 0);
    }

    #[test]
    fn executor_empty_union() {
        use crate::plan::physical::Cost;

        // Empty Union (no inputs)
        let plan = PhysicalPlan::Union { all: false, cost: Cost::default(), inputs: vec![] };

        let ctx = ExecutionContext::new();
        let mut executor = Executor::new(&plan, ctx).unwrap();
        assert_eq!(executor.count().unwrap(), 0);
    }
}
