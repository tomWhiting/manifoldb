//! Physical plan builder.
//!
//! This module converts logical plans into physical plans by choosing
//! concrete algorithms and access methods.

// Allow matching arms with identical bodies - intentional for grouping
#![allow(clippy::match_same_arms)]
// Allow cognitive complexity for the main planning function
#![allow(clippy::cognitive_complexity)]
// Allow too many lines for the comprehensive planner
#![allow(clippy::too_many_lines)]

use super::cost::{Cost, CostModel};
use super::node::{
    AnalyzeExecNode, BruteForceSearchNode, CallSubqueryExecNode, CopyExecFormat, CopyExecNode,
    CteCycleExecConfig, CteSearchExecConfig, ExplainAnalyzeExecNode, ExplainExecFormat,
    FilterExecNode, FullScanNode, GraphExpandExecNode, GraphPathScanExecNode, HashAggregateNode,
    HashJoinNode, HnswSearchNode, HybridSearchComponentNode,
    HybridSearchNode as PhysicalHybridSearchNode, IndexRangeScanNode, IndexScanNode, JoinOrder,
    LimitExecNode, NestedLoopJoinNode, PhysicalPlan, PhysicalScoreCombinationMethod,
    ProjectExecNode, RecursiveCTEExecNode, ResetExecNode, SetSessionExecNode, ShortestPathExecNode,
    ShowExecNode, ShowProceduresExecNode, SortExecNode, UnwindExecNode, VacuumExecNode,
    WindowExecNode, WindowFunctionExpr,
};
use crate::plan::logical::{
    AggregateNode, AnnSearchNode, CteSearchOrder, ExpandNode, HybridSearchNode, JoinNode, JoinType,
    LogicalExpr, LogicalPlan, PathScanNode, RecursiveCTENode, ScanNode, ScoreCombinationMethod,
    ShortestPathNode, UnwindNode,
};
use crate::plan::optimize::{AccessType, IndexCandidate, IndexSelector};

/// Physical query planner.
///
/// Transforms logical plans into physical plans by:
/// 1. Choosing concrete algorithms (e.g., hash join vs nested loop)
/// 2. Selecting access methods (e.g., index scan vs full scan)
/// 3. Estimating costs for optimizer decisions
///
/// # Example
///
/// ```ignore
/// use manifoldb_query::plan::physical::PhysicalPlanner;
/// use manifoldb_query::plan::LogicalPlan;
///
/// let logical = LogicalPlan::scan("users")
///     .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));
///
/// let planner = PhysicalPlanner::new();
/// let physical = planner.plan(&logical)?;
/// ```
#[derive(Debug, Clone)]
pub struct PhysicalPlanner {
    /// Cost model for estimating operator costs.
    cost_model: CostModel,
    /// Catalog information for index selection.
    catalog: PlannerCatalog,
}

/// Catalog information for the planner.
///
/// Provides table statistics and index information for optimization.
#[derive(Debug, Clone, Default)]
pub struct PlannerCatalog {
    /// Table statistics.
    tables: Vec<TableStats>,
    /// Available indexes.
    indexes: Vec<IndexInfo>,
}

/// Statistics for a table.
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Table name.
    pub name: String,
    /// Estimated row count.
    pub row_count: usize,
    /// Average row size in bytes.
    pub avg_row_size: usize,
}

impl TableStats {
    /// Creates new table statistics.
    #[must_use]
    pub fn new(name: impl Into<String>, row_count: usize) -> Self {
        Self {
            name: name.into(),
            row_count,
            avg_row_size: 100, // Default estimate
        }
    }

    /// Sets the average row size.
    #[must_use]
    pub const fn with_avg_row_size(mut self, size: usize) -> Self {
        self.avg_row_size = size;
        self
    }
}

/// Information about an available index.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Index name.
    pub name: String,
    /// Table the index is on.
    pub table: String,
    /// Columns in the index (in order).
    pub columns: Vec<String>,
    /// Whether this is a unique index.
    pub unique: bool,
    /// Index type.
    pub index_type: IndexType,
    /// Collection name for named vector indexes.
    pub collection_name: Option<String>,
    /// Vector name for named vector indexes.
    pub vector_name: Option<String>,
}

impl IndexInfo {
    /// Creates a new B-tree index info.
    #[must_use]
    pub fn btree(name: impl Into<String>, table: impl Into<String>, columns: Vec<String>) -> Self {
        Self {
            name: name.into(),
            table: table.into(),
            columns,
            unique: false,
            index_type: IndexType::BTree,
            collection_name: None,
            vector_name: None,
        }
    }

    /// Creates a new unique B-tree index info.
    #[must_use]
    pub fn unique_btree(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            table: table.into(),
            columns,
            unique: true,
            index_type: IndexType::BTree,
            collection_name: None,
            vector_name: None,
        }
    }

    /// Creates a new HNSW vector index info.
    #[must_use]
    pub fn hnsw(
        name: impl Into<String>,
        table: impl Into<String>,
        column: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            table: table.into(),
            columns: vec![column.into()],
            unique: false,
            index_type: IndexType::Hnsw,
            collection_name: None,
            vector_name: None,
        }
    }

    /// Creates a new HNSW vector index info for a named vector in a collection.
    ///
    /// The index name follows the convention: `{collection}_{vector_name}_hnsw`
    #[must_use]
    pub fn hnsw_for_named_vector(
        collection: impl Into<String>,
        vector_name: impl Into<String>,
    ) -> Self {
        let collection = collection.into();
        let vector = vector_name.into();
        let index_name = format!("{collection}_{vector}_hnsw");
        Self {
            name: index_name,
            table: collection.clone(),
            columns: vec![vector.clone()],
            unique: false,
            index_type: IndexType::Hnsw,
            collection_name: Some(collection),
            vector_name: Some(vector),
        }
    }
}

/// Index type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// B-tree index (for equality and range queries).
    BTree,
    /// Hash index (for equality queries only).
    Hash,
    /// HNSW vector index.
    Hnsw,
}

impl PlannerCatalog {
    /// Creates an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds table statistics.
    #[must_use]
    pub fn with_table(mut self, stats: TableStats) -> Self {
        self.tables.push(stats);
        self
    }

    /// Adds an index.
    #[must_use]
    pub fn with_index(mut self, index: IndexInfo) -> Self {
        self.indexes.push(index);
        self
    }

    /// Gets statistics for a table.
    #[must_use]
    pub fn get_table_stats(&self, table: &str) -> Option<&TableStats> {
        self.tables.iter().find(|t| t.name == table)
    }

    /// Gets the estimated row count for a table.
    #[must_use]
    pub fn get_row_count(&self, table: &str) -> usize {
        self.get_table_stats(table).map_or(10_000, |s| s.row_count) // Default estimate
    }

    /// Finds indexes for a table.
    #[must_use]
    pub fn get_indexes(&self, table: &str) -> Vec<&IndexInfo> {
        self.indexes.iter().filter(|i| i.table == table).collect()
    }

    /// Finds an HNSW index for a column.
    #[must_use]
    pub fn get_hnsw_index(&self, table: &str, column: &str) -> Option<&IndexInfo> {
        self.indexes.iter().find(|i| {
            i.table == table
                && i.index_type == IndexType::Hnsw
                && i.columns.first().is_some_and(|c| c == column)
        })
    }

    /// Finds an HNSW index for a named vector in a collection.
    ///
    /// This looks for indexes that were created with collection and vector metadata,
    /// following the naming convention `{collection}_{vector_name}_hnsw`.
    #[must_use]
    pub fn get_hnsw_index_for_named_vector(
        &self,
        collection: &str,
        vector_name: &str,
    ) -> Option<&IndexInfo> {
        // First try to find by explicit collection/vector metadata
        self.indexes
            .iter()
            .find(|i| {
                i.index_type == IndexType::Hnsw
                    && i.collection_name.as_deref() == Some(collection)
                    && i.vector_name.as_deref() == Some(vector_name)
            })
            .or_else(|| {
                // Fall back to naming convention matching
                let expected_name = format!("{collection}_{vector_name}_hnsw");
                self.indexes
                    .iter()
                    .find(|i| i.index_type == IndexType::Hnsw && i.name == expected_name)
            })
    }

    /// Checks if an HNSW index exists for a named vector.
    #[must_use]
    pub fn has_hnsw_index_for_named_vector(&self, collection: &str, vector_name: &str) -> bool {
        self.get_hnsw_index_for_named_vector(collection, vector_name).is_some()
    }

    /// Finds a B-tree index that can serve a predicate.
    #[must_use]
    pub fn find_btree_index(&self, table: &str, column: &str) -> Option<&IndexInfo> {
        self.indexes.iter().find(|i| {
            i.table == table
                && matches!(i.index_type, IndexType::BTree | IndexType::Hash)
                && i.columns.first().is_some_and(|c| c == column)
        })
    }

    /// Merge another catalog into this one.
    ///
    /// Indexes and tables from the other catalog are added to this catalog.
    /// Duplicates are not checked - if the same index exists in both, it will appear twice.
    #[must_use]
    pub fn merge(mut self, other: PlannerCatalog) -> Self {
        self.tables.extend(other.tables);
        self.indexes.extend(other.indexes);
        self
    }
}

impl Default for PhysicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalPlanner {
    /// Creates a new physical planner with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { cost_model: CostModel::new(), catalog: PlannerCatalog::new() }
    }

    /// Sets the cost model.
    #[must_use]
    pub fn with_cost_model(mut self, model: CostModel) -> Self {
        self.cost_model = model;
        self
    }

    /// Sets the catalog.
    #[must_use]
    pub fn with_catalog(mut self, catalog: PlannerCatalog) -> Self {
        self.catalog = catalog;
        self
    }

    /// Returns a reference to the cost model.
    #[must_use]
    pub const fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// Returns a reference to the catalog.
    #[must_use]
    pub const fn catalog(&self) -> &PlannerCatalog {
        &self.catalog
    }

    /// Converts a logical plan to a physical plan.
    #[must_use]
    pub fn plan(&self, logical: &LogicalPlan) -> PhysicalPlan {
        match logical {
            // Leaf nodes
            LogicalPlan::Scan(node) => self.plan_scan(node),
            LogicalPlan::Values(node) => self.plan_values(node),
            LogicalPlan::Empty { columns } => PhysicalPlan::Empty { columns: columns.clone() },

            // Unary nodes
            LogicalPlan::Filter { node, input } => self.plan_filter(node, input),
            LogicalPlan::Project { node, input } => self.plan_project(node, input),
            LogicalPlan::Aggregate { node, input } => self.plan_aggregate(node, input),
            LogicalPlan::Sort { node, input } => self.plan_sort(node, input),
            LogicalPlan::Limit { node, input } => self.plan_limit(node, input),
            LogicalPlan::Distinct { node, input } => self.plan_distinct(node, input),
            LogicalPlan::Window { node, input } => self.plan_window(node, input),
            LogicalPlan::Alias { input, .. } => {
                // Alias is logical-only, just plan the input
                self.plan(input)
            }
            LogicalPlan::Unwind { node, input } => self.plan_unwind(node, input),

            // Binary nodes
            LogicalPlan::Join { node, left, right } => self.plan_join(node, left, right),
            LogicalPlan::SetOp { node, left, right } => self.plan_set_op(node, left, right),

            // N-ary nodes
            LogicalPlan::Union { node, inputs } => self.plan_union(node, inputs),

            // Recursive nodes
            LogicalPlan::RecursiveCTE { node, initial, recursive } => {
                self.plan_recursive_cte(node, initial, recursive)
            }

            // Graph nodes
            LogicalPlan::Expand { node, input } => self.plan_expand(node, input),
            LogicalPlan::PathScan { node, input } => self.plan_path_scan(node, input),
            LogicalPlan::ShortestPath { node, input } => self.plan_shortest_path(node, input),

            // Vector nodes
            LogicalPlan::AnnSearch { node, input } => self.plan_ann_search(node, input),
            LogicalPlan::VectorDistance { node, input } => self.plan_vector_distance(node, input),
            LogicalPlan::HybridSearch { node, input } => self.plan_hybrid_search(node, input),

            // DML nodes
            LogicalPlan::Insert { table, columns, input, on_conflict, returning } => {
                self.plan_insert(table, columns, input, on_conflict, returning)
            }
            LogicalPlan::Update { table, assignments, filter, returning } => {
                self.plan_update(table, assignments, filter, returning)
            }
            LogicalPlan::Delete { table, filter, returning } => {
                self.plan_delete(table, filter, returning)
            }
            LogicalPlan::MergeSql { target_table, source, on_condition, clauses } => {
                self.plan_merge_sql(target_table, source, on_condition, clauses)
            }

            // DDL nodes - these are executed directly without physical plans
            LogicalPlan::CreateTable(node) => PhysicalPlan::CreateTable(node.clone()),
            LogicalPlan::AlterTable(node) => PhysicalPlan::AlterTable(node.clone()),
            LogicalPlan::DropTable(node) => PhysicalPlan::DropTable(node.clone()),
            LogicalPlan::TruncateTable(node) => PhysicalPlan::TruncateTable(node.clone()),
            LogicalPlan::CreateIndex(node) => PhysicalPlan::CreateIndex(node.clone()),
            LogicalPlan::AlterIndex(node) => PhysicalPlan::AlterIndex(node.clone()),
            LogicalPlan::DropIndex(node) => PhysicalPlan::DropIndex(node.clone()),
            LogicalPlan::CreateCollection(node) => PhysicalPlan::CreateCollection(node.clone()),
            LogicalPlan::DropCollection(node) => PhysicalPlan::DropCollection(node.clone()),
            LogicalPlan::CreateView(node) => PhysicalPlan::CreateView(node.clone()),
            LogicalPlan::DropView(node) => PhysicalPlan::DropView(node.clone()),
            LogicalPlan::CreateSchema(node) => PhysicalPlan::CreateSchema(node.clone()),
            LogicalPlan::AlterSchema(node) => PhysicalPlan::AlterSchema(node.clone()),
            LogicalPlan::DropSchema(node) => PhysicalPlan::DropSchema(node.clone()),
            LogicalPlan::CreateFunction(node) => PhysicalPlan::CreateFunction((**node).clone()),
            LogicalPlan::DropFunction(node) => PhysicalPlan::DropFunction(node.clone()),
            LogicalPlan::CreateTrigger(node) => PhysicalPlan::CreateTrigger((**node).clone()),
            LogicalPlan::DropTrigger(node) => PhysicalPlan::DropTrigger(node.clone()),

            // Graph DML nodes - build physical plans for these
            LogicalPlan::GraphCreate { node, input } => {
                let input_plan = input.as_ref().map(|i| Box::new(self.plan(i)));
                PhysicalPlan::GraphCreate { node: node.clone(), input: input_plan }
            }
            LogicalPlan::GraphMerge { node, input } => {
                let input_plan = input.as_ref().map(|i| Box::new(self.plan(i)));
                PhysicalPlan::GraphMerge { node: node.clone(), input: input_plan }
            }
            LogicalPlan::GraphSet { node, input } => {
                let input_plan = Box::new(self.plan(input));
                PhysicalPlan::GraphSet { node: node.clone(), input: input_plan }
            }
            LogicalPlan::GraphDelete { node, input } => {
                let input_plan = Box::new(self.plan(input));
                PhysicalPlan::GraphDelete { node: node.clone(), input: input_plan }
            }
            LogicalPlan::GraphRemove { node, input } => {
                let input_plan = Box::new(self.plan(input));
                PhysicalPlan::GraphRemove { node: node.clone(), input: input_plan }
            }
            LogicalPlan::GraphForeach { node, input } => {
                let input_plan = Box::new(self.plan(input));
                PhysicalPlan::GraphForeach { node: node.clone(), input: input_plan }
            }
            LogicalPlan::ProcedureCall(node) => PhysicalPlan::ProcedureCall(node.clone()),

            // CALL { } subquery
            LogicalPlan::CallSubquery { node, subquery, input } => {
                let input_plan = Box::new(self.plan(input));
                let subquery_plan = Box::new(self.plan(subquery));
                let exec_node =
                    Box::new(CallSubqueryExecNode::new(node.imported_variables.clone()));
                PhysicalPlan::CallSubquery {
                    node: exec_node,
                    subquery: subquery_plan,
                    input: input_plan,
                }
            }

            // Transaction control nodes - pass through directly
            LogicalPlan::BeginTransaction(node) => PhysicalPlan::BeginTransaction(node.clone()),
            LogicalPlan::Commit(node) => PhysicalPlan::Commit(node.clone()),
            LogicalPlan::Rollback(node) => PhysicalPlan::Rollback(node.clone()),
            LogicalPlan::Savepoint(node) => PhysicalPlan::Savepoint(node.clone()),
            LogicalPlan::ReleaseSavepoint(node) => PhysicalPlan::ReleaseSavepoint(node.clone()),
            LogicalPlan::SetTransaction(node) => PhysicalPlan::SetTransaction(node.clone()),

            // Utility statements - convert logical to physical nodes
            LogicalPlan::ExplainAnalyze(node) => {
                let input_plan = Box::new(self.plan(&node.input));
                PhysicalPlan::ExplainAnalyze(Box::new(ExplainAnalyzeExecNode {
                    input: input_plan,
                    buffers: node.buffers,
                    timing: node.timing,
                    format: match node.format {
                        crate::plan::logical::ExplainFormat::Text => ExplainExecFormat::Text,
                        crate::plan::logical::ExplainFormat::Json => ExplainExecFormat::Json,
                        crate::plan::logical::ExplainFormat::Xml => ExplainExecFormat::Xml,
                        crate::plan::logical::ExplainFormat::Yaml => ExplainExecFormat::Yaml,
                    },
                    verbose: node.verbose,
                    costs: node.costs,
                }))
            }
            LogicalPlan::Vacuum(node) => PhysicalPlan::Vacuum(VacuumExecNode {
                full: node.full,
                analyze: node.analyze,
                table: node.table.as_ref().map(|t| t.to_string()),
                columns: node.columns.clone(),
            }),
            LogicalPlan::Analyze(node) => PhysicalPlan::Analyze(AnalyzeExecNode {
                table: node.table.as_ref().map(|t| t.to_string()),
                columns: node.columns.clone(),
            }),
            LogicalPlan::Copy(node) => {
                use crate::ast::{CopyDestination, CopyDirection, CopySource, CopyTarget};

                // Extract table name and columns from target
                let (table, columns) = match &node.target {
                    CopyTarget::Table { name, columns } => {
                        (Some(name.to_string()), columns.iter().map(|c| c.to_string()).collect())
                    }
                    CopyTarget::Query(_) => (None, vec![]),
                };

                // Extract file path from direction
                let path = match &node.direction {
                    CopyDirection::To(dest) => match dest {
                        CopyDestination::File(p) => Some(p.clone()),
                        _ => None,
                    },
                    CopyDirection::From(src) => match src {
                        CopySource::File(p) => Some(p.clone()),
                        _ => None,
                    },
                };

                // Convert format
                let format = match node.options.format {
                    crate::ast::CopyFormat::Csv => CopyExecFormat::Csv,
                    crate::ast::CopyFormat::Text => CopyExecFormat::Text,
                    crate::ast::CopyFormat::Binary => CopyExecFormat::Binary,
                };

                PhysicalPlan::Copy(CopyExecNode {
                    is_export: node.is_export(),
                    table,
                    columns,
                    path,
                    header: node.options.header,
                    delimiter: node.options.delimiter,
                    format,
                })
            }
            LogicalPlan::SetSession(node) => PhysicalPlan::SetSession(SetSessionExecNode {
                name: node.name.clone(),
                value: node.value.as_ref().map(|v| v.to_string()),
                local: node.local,
            }),
            LogicalPlan::Show(node) => PhysicalPlan::Show(ShowExecNode { name: node.name.clone() }),
            LogicalPlan::Reset(node) => {
                PhysicalPlan::Reset(ResetExecNode { name: node.name.clone() })
            }
            LogicalPlan::ShowProcedures(node) => {
                PhysicalPlan::ShowProcedures(ShowProceduresExecNode { executable: node.executable })
            }
        }
    }

    fn plan_scan(&self, node: &ScanNode) -> PhysicalPlan {
        let row_count = self.catalog.get_row_count(&node.table_name);

        // Try to use an index if there's a filter predicate
        if let Some(filter) = &node.filter {
            if let Some(plan) = self.try_index_scan(node, filter, row_count) {
                return plan;
            }
        }

        // Fall back to full table scan
        let cost = self.cost_model.full_scan_cost(row_count);
        let mut scan = FullScanNode::new(&node.table_name).with_cost(cost);

        if let Some(alias) = &node.alias {
            scan = scan.with_alias(alias);
        }

        if let Some(proj) = &node.projection {
            scan = scan.with_projection(proj.clone());
        }

        if let Some(filter) = &node.filter {
            scan = scan.with_filter(filter.clone());
        }

        PhysicalPlan::FullScan(Box::new(scan))
    }

    /// Tries to create an index scan plan for the given filter predicate.
    ///
    /// Returns `Some(PhysicalPlan)` if an index can be used, `None` otherwise.
    fn try_index_scan(
        &self,
        node: &ScanNode,
        filter: &LogicalExpr,
        row_count: usize,
    ) -> Option<PhysicalPlan> {
        // Use IndexSelector to find index candidates
        let selector = IndexSelector::new();
        let candidates = selector.find_index_candidates(filter);

        if candidates.is_empty() {
            return None;
        }

        // Find the best candidate that has an available index
        let mut best_plan: Option<(Cost, PhysicalPlan)> = None;

        for candidate in &candidates {
            // Check if we have an index for this column
            if let Some(index_info) =
                self.catalog.find_btree_index(&node.table_name, &candidate.column)
            {
                // Calculate index scan cost
                let (plan, index_cost) =
                    self.build_index_plan(node, filter, candidate, &index_info.name, row_count);

                // Compare with current best
                if let Some((ref current_cost, _)) = best_plan {
                    if index_cost.is_less_than(current_cost) {
                        best_plan = Some((index_cost, plan));
                    }
                } else {
                    best_plan = Some((index_cost, plan));
                }
            }
        }

        // Compare index plan cost with full scan cost
        if let Some((index_cost, plan)) = best_plan {
            let full_scan_cost = self.cost_model.full_scan_cost(row_count);
            if index_cost.is_less_than(&full_scan_cost) {
                return Some(plan);
            }
        }

        None
    }

    /// Builds an index scan plan for the given candidate.
    fn build_index_plan(
        &self,
        node: &ScanNode,
        filter: &LogicalExpr,
        candidate: &IndexCandidate,
        index_name: &str,
        row_count: usize,
    ) -> (PhysicalPlan, Cost) {
        match candidate.access_type {
            AccessType::PointLookup => {
                self.build_point_lookup_plan(node, filter, candidate, index_name, row_count)
            }
            AccessType::RangeScan | AccessType::RangeScanGt | AccessType::RangeScanLt => {
                self.build_range_scan_plan(node, filter, candidate, index_name, row_count)
            }
            AccessType::InList => {
                // IN lists are handled as multiple point lookups; for now fall back to range
                self.build_range_scan_plan(node, filter, candidate, index_name, row_count)
            }
            AccessType::PrefixScan => {
                // Prefix scans are handled as range scans
                self.build_range_scan_plan(node, filter, candidate, index_name, row_count)
            }
        }
    }

    /// Builds a point lookup (equality) index scan plan.
    fn build_point_lookup_plan(
        &self,
        node: &ScanNode,
        filter: &LogicalExpr,
        candidate: &IndexCandidate,
        index_name: &str,
        row_count: usize,
    ) -> (PhysicalPlan, Cost) {
        // Extract the lookup value from the filter predicate
        let key_value = self.extract_equality_value(filter, &candidate.column);

        // Estimate result count (usually 1 for unique, small for non-unique)
        let result_count = (row_count as f64 * candidate.selectivity).ceil() as usize;
        let cost = self.cost_model.index_lookup_cost(result_count.max(1));

        let mut scan_node = IndexScanNode::new(
            &node.table_name,
            index_name,
            vec![candidate.column.clone()],
            vec![key_value.unwrap_or_else(|| LogicalExpr::null())],
        )
        .with_cost(cost);

        if let Some(proj) = &node.projection {
            scan_node = scan_node.with_projection(proj.clone());
        }

        // Check if there are residual predicates not covered by the index
        let residual = self.extract_residual_predicate(filter, &candidate.column);

        let plan = if let Some(residual_pred) = residual {
            // Need to apply residual filter on top of index scan
            let filter_cost = self.cost_model.filter_cost(result_count, 0.5);
            PhysicalPlan::Filter {
                node: FilterExecNode::new(residual_pred).with_cost(filter_cost),
                input: Box::new(PhysicalPlan::IndexScan(Box::new(scan_node))),
            }
        } else {
            PhysicalPlan::IndexScan(Box::new(scan_node))
        };

        (plan, cost)
    }

    /// Builds a range scan index plan.
    fn build_range_scan_plan(
        &self,
        node: &ScanNode,
        filter: &LogicalExpr,
        candidate: &IndexCandidate,
        index_name: &str,
        row_count: usize,
    ) -> (PhysicalPlan, Cost) {
        // Extract bounds from the filter predicate
        let (lower_bound, lower_inclusive, upper_bound, upper_inclusive) =
            self.extract_range_bounds(filter, &candidate.column);

        let cost = self.cost_model.index_range_cost(row_count, candidate.selectivity);
        let result_count = (row_count as f64 * candidate.selectivity).ceil() as usize;

        let scan_node = IndexRangeScanNode {
            table_name: node.table_name.clone(),
            index_name: index_name.to_string(),
            key_column: candidate.column.clone(),
            lower_bound,
            lower_inclusive,
            upper_bound,
            upper_inclusive,
            projection: node.projection.clone(),
            cost,
        };

        // Check if there are residual predicates
        let residual = self.extract_residual_predicate(filter, &candidate.column);

        let plan = if let Some(residual_pred) = residual {
            let filter_cost = self.cost_model.filter_cost(result_count, 0.5);
            PhysicalPlan::Filter {
                node: FilterExecNode::new(residual_pred).with_cost(filter_cost),
                input: Box::new(PhysicalPlan::IndexRangeScan(Box::new(scan_node))),
            }
        } else {
            PhysicalPlan::IndexRangeScan(Box::new(scan_node))
        };

        (plan, cost)
    }

    /// Extracts the equality value from a filter predicate for a specific column.
    fn extract_equality_value(&self, filter: &LogicalExpr, column: &str) -> Option<LogicalExpr> {
        match filter {
            LogicalExpr::BinaryOp { left, op: crate::ast::BinaryOp::Eq, right } => {
                // column = value
                if self.is_column(left, column) {
                    return Some(*right.clone());
                }
                // value = column
                if self.is_column(right, column) {
                    return Some(*left.clone());
                }
                None
            }
            LogicalExpr::BinaryOp { left, op: crate::ast::BinaryOp::And, right } => {
                // Try both sides of AND
                self.extract_equality_value(left, column)
                    .or_else(|| self.extract_equality_value(right, column))
            }
            _ => None,
        }
    }

    /// Extracts range bounds from a filter predicate.
    fn extract_range_bounds(
        &self,
        filter: &LogicalExpr,
        column: &str,
    ) -> (Option<LogicalExpr>, bool, Option<LogicalExpr>, bool) {
        let mut lower_bound = None;
        let mut lower_inclusive = false;
        let mut upper_bound = None;
        let mut upper_inclusive = false;

        self.collect_range_bounds(
            filter,
            column,
            &mut lower_bound,
            &mut lower_inclusive,
            &mut upper_bound,
            &mut upper_inclusive,
        );

        (lower_bound, lower_inclusive, upper_bound, upper_inclusive)
    }

    fn collect_range_bounds(
        &self,
        filter: &LogicalExpr,
        column: &str,
        lower_bound: &mut Option<LogicalExpr>,
        lower_inclusive: &mut bool,
        upper_bound: &mut Option<LogicalExpr>,
        upper_inclusive: &mut bool,
    ) {
        match filter {
            LogicalExpr::BinaryOp { left, op, right } => {
                match op {
                    crate::ast::BinaryOp::Gt => {
                        if self.is_column(left, column) {
                            *lower_bound = Some(*right.clone());
                            *lower_inclusive = false;
                        } else if self.is_column(right, column) {
                            *upper_bound = Some(*left.clone());
                            *upper_inclusive = false;
                        }
                    }
                    crate::ast::BinaryOp::GtEq => {
                        if self.is_column(left, column) {
                            *lower_bound = Some(*right.clone());
                            *lower_inclusive = true;
                        } else if self.is_column(right, column) {
                            *upper_bound = Some(*left.clone());
                            *upper_inclusive = true;
                        }
                    }
                    crate::ast::BinaryOp::Lt => {
                        if self.is_column(left, column) {
                            *upper_bound = Some(*right.clone());
                            *upper_inclusive = false;
                        } else if self.is_column(right, column) {
                            *lower_bound = Some(*left.clone());
                            *lower_inclusive = false;
                        }
                    }
                    crate::ast::BinaryOp::LtEq => {
                        if self.is_column(left, column) {
                            *upper_bound = Some(*right.clone());
                            *upper_inclusive = true;
                        } else if self.is_column(right, column) {
                            *lower_bound = Some(*left.clone());
                            *lower_inclusive = true;
                        }
                    }
                    crate::ast::BinaryOp::And => {
                        // Recurse into both sides
                        self.collect_range_bounds(
                            left,
                            column,
                            lower_bound,
                            lower_inclusive,
                            upper_bound,
                            upper_inclusive,
                        );
                        self.collect_range_bounds(
                            right,
                            column,
                            lower_bound,
                            lower_inclusive,
                            upper_bound,
                            upper_inclusive,
                        );
                    }
                    _ => {}
                }
            }
            LogicalExpr::Between { expr, low, high, negated } if !negated => {
                if self.is_column(expr, column) {
                    *lower_bound = Some(*low.clone());
                    *lower_inclusive = true;
                    *upper_bound = Some(*high.clone());
                    *upper_inclusive = true;
                }
            }
            _ => {}
        }
    }

    /// Checks if an expression is a reference to the specified column.
    fn is_column(&self, expr: &LogicalExpr, column: &str) -> bool {
        match expr {
            LogicalExpr::Column { name, .. } => name == column,
            _ => false,
        }
    }

    /// Extracts predicates that are not covered by the index lookup.
    fn extract_residual_predicate(
        &self,
        filter: &LogicalExpr,
        indexed_column: &str,
    ) -> Option<LogicalExpr> {
        match filter {
            LogicalExpr::BinaryOp { left, op: crate::ast::BinaryOp::And, right } => {
                let left_residual = self.extract_residual_predicate(left, indexed_column);
                let right_residual = self.extract_residual_predicate(right, indexed_column);

                match (left_residual, right_residual) {
                    (Some(l), Some(r)) => Some(l.and(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            LogicalExpr::BinaryOp { left, op, right } => {
                // Check if this predicate uses the indexed column
                let uses_indexed =
                    self.is_column(left, indexed_column) || self.is_column(right, indexed_column);

                if uses_indexed {
                    // This predicate is handled by the index
                    match op {
                        crate::ast::BinaryOp::Eq
                        | crate::ast::BinaryOp::Lt
                        | crate::ast::BinaryOp::LtEq
                        | crate::ast::BinaryOp::Gt
                        | crate::ast::BinaryOp::GtEq => None,
                        _ => Some(filter.clone()), // Other ops need residual check
                    }
                } else {
                    // This predicate is not covered by the index
                    Some(filter.clone())
                }
            }
            LogicalExpr::Between { expr, negated, .. } if !negated => {
                if self.is_column(expr, indexed_column) {
                    None // BETWEEN is handled by index
                } else {
                    Some(filter.clone())
                }
            }
            _ => Some(filter.clone()), // Unknown predicates need residual check
        }
    }

    fn plan_values(&self, node: &crate::plan::logical::ValuesNode) -> PhysicalPlan {
        let row_count = node.rows.len();
        let cost = Cost::new(row_count as f64 * 0.1, row_count);

        PhysicalPlan::Values { rows: node.rows.clone(), cost }
    }

    fn plan_filter(
        &self,
        node: &crate::plan::logical::FilterNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_cardinality = input_plan.cost().cardinality();

        // Estimate selectivity (default 50%)
        let selectivity = self.estimate_selectivity(&node.predicate);
        let cost = self.cost_model.filter_cost(input_cardinality, selectivity);

        PhysicalPlan::Filter {
            node: FilterExecNode::new(node.predicate.clone())
                .with_selectivity(selectivity)
                .with_cost(cost),
            input: Box::new(input_plan),
        }
    }

    fn plan_project(
        &self,
        node: &crate::plan::logical::ProjectNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let row_count = input_plan.cost().cardinality();
        let cost = self.cost_model.project_cost(row_count, node.exprs.len());

        PhysicalPlan::Project {
            node: ProjectExecNode::new(node.exprs.clone()).with_cost(cost),
            input: Box::new(input_plan),
        }
    }

    fn plan_aggregate(&self, node: &AggregateNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_rows = input_plan.cost().cardinality();

        // Estimate output groups - for grouping sets, multiply by number of sets
        let base_group_count = if node.group_by.is_empty() {
            1 // Simple aggregation returns one row
        } else {
            // Estimate distinct values (rough heuristic)
            (input_rows as f64).sqrt().ceil() as usize
        };

        let group_count = if node.has_grouping_sets() {
            // Each grouping set produces its own set of groups
            base_group_count * node.grouping_sets.len().max(1)
        } else {
            base_group_count
        };

        let cost = self.cost_model.hash_aggregate_cost(input_rows, group_count);

        let mut agg_node = if node.has_grouping_sets() {
            HashAggregateNode::with_grouping_sets(
                node.group_by.clone(),
                node.aggregates.clone(),
                node.grouping_sets.clone(),
            )
            .with_cost(cost)
        } else {
            HashAggregateNode::new(node.group_by.clone(), node.aggregates.clone()).with_cost(cost)
        };

        if let Some(having) = &node.having {
            agg_node = agg_node.with_having(having.clone());
        }

        PhysicalPlan::HashAggregate { node: Box::new(agg_node), input: Box::new(input_plan) }
    }

    fn plan_sort(
        &self,
        node: &crate::plan::logical::SortNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let row_count = input_plan.cost().cardinality();
        let cost = self.cost_model.sort_cost(row_count);

        PhysicalPlan::Sort {
            node: SortExecNode::new(node.order_by.clone()).with_cost(cost),
            input: Box::new(input_plan),
        }
    }

    fn plan_limit(
        &self,
        node: &crate::plan::logical::LimitNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);

        PhysicalPlan::Limit {
            node: LimitExecNode { limit: node.limit, offset: node.offset },
            input: Box::new(input_plan),
        }
    }

    fn plan_distinct(
        &self,
        node: &crate::plan::logical::DistinctNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_rows = input_plan.cost().cardinality();

        // Estimate distinct count (rough heuristic)
        let distinct_count = (input_rows as f64 * 0.8).ceil() as usize;
        let cost = self.cost_model.hash_distinct_cost(input_rows, distinct_count);

        PhysicalPlan::HashDistinct {
            on_columns: node.on_columns.clone(),
            cost,
            input: Box::new(input_plan),
        }
    }

    fn plan_unwind(&self, node: &UnwindNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_rows = input_plan.cost().cardinality();

        // Estimate average list size (assume ~5 elements per list on average)
        let avg_list_size = 5;
        let output_rows = input_rows * avg_list_size;
        let cost = Cost::new(output_rows as f64, output_rows);

        PhysicalPlan::Unwind {
            node: UnwindExecNode::new(node.list_expr.clone(), &node.alias).with_cost(cost),
            input: Box::new(input_plan),
        }
    }

    fn plan_recursive_cte(
        &self,
        node: &RecursiveCTENode,
        initial: &LogicalPlan,
        recursive: &LogicalPlan,
    ) -> PhysicalPlan {
        let initial_plan = self.plan(initial);
        let recursive_plan = self.plan(recursive);

        let initial_rows = initial_plan.cost().cardinality();

        // Estimate output: assume recursive query produces decreasing rows per iteration
        // This is a rough estimate - actual rows depend on data characteristics
        // Estimate average depth of ~5 iterations for hierarchical queries
        let avg_iterations = 5;
        let output_rows = initial_rows * avg_iterations;

        // Cost: initial execution + (iterations * recursive execution)
        // Plus overhead for working table management
        let cost = Cost::new(
            initial_plan.cost().value()
                + (avg_iterations as f64 * recursive_plan.cost().value())
                + (output_rows as f64 * 0.1), // working table overhead
            output_rows,
        );

        let max_iterations =
            node.max_iterations.unwrap_or(RecursiveCTEExecNode::DEFAULT_MAX_ITERATIONS);

        // Convert search config from logical to physical
        let search_config = node.search_config.as_ref().map(|config| {
            // Resolve column names to indices in the CTE output
            let by_column_indices: Vec<usize> = config
                .by_columns
                .iter()
                .filter_map(|col| node.columns.iter().position(|c| c == col))
                .collect();
            // The set column is added at the end
            let set_column_index = node.columns.len();
            CteSearchExecConfig::new(
                matches!(config.order, CteSearchOrder::DepthFirst),
                by_column_indices,
                set_column_index,
            )
        });

        // Convert cycle config from logical to physical
        let cycle_config = node.cycle_config.as_ref().map(|config| {
            // Resolve column names to indices in the CTE output
            let column_indices: Vec<usize> = config
                .columns
                .iter()
                .filter_map(|col| node.columns.iter().position(|c| c == col))
                .collect();
            // The mark column is added at the end (after search column if present)
            let mark_column_index =
                node.columns.len() + if node.search_config.is_some() { 1 } else { 0 };
            // Path column is after mark column if present
            let path_column_index = config.path_column.as_ref().map(|_| mark_column_index + 1);
            CteCycleExecConfig { column_indices, mark_column_index, path_column_index }
        });

        let mut exec_node =
            RecursiveCTEExecNode::new(&node.name, node.columns.clone(), node.union_all)
                .with_max_iterations(max_iterations)
                .with_cost(cost);

        if let Some(search) = search_config {
            exec_node = exec_node.with_search(search);
        }
        if let Some(cycle) = cycle_config {
            exec_node = exec_node.with_cycle(cycle);
        }

        PhysicalPlan::RecursiveCTE {
            node: Box::new(exec_node),
            initial: Box::new(initial_plan),
            recursive: Box::new(recursive_plan),
        }
    }

    fn plan_window(
        &self,
        node: &crate::plan::logical::WindowNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_rows = input_plan.cost().cardinality();

        // Convert logical window expressions to physical
        let window_exprs: Vec<WindowFunctionExpr> = node
            .window_exprs
            .iter()
            .filter_map(|(expr, alias)| {
                if let LogicalExpr::WindowFunction {
                    func,
                    arg,
                    default_value,
                    partition_by,
                    order_by,
                    frame,
                    filter,
                } = expr
                {
                    Some(WindowFunctionExpr::with_frame(
                        func.clone(),
                        arg.as_ref().map(|a| (**a).clone()),
                        default_value.as_ref().map(|d| (**d).clone()),
                        partition_by.clone(),
                        order_by.clone(),
                        frame.clone(),
                        filter.as_ref().map(|f| (**f).clone()),
                        alias.clone(),
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Window function cost: O(n log n) for sorting + O(n) for computation
        let cost = Cost::new(
            (input_rows as f64) * (input_rows as f64).log2() + (input_rows as f64),
            input_rows,
        );

        PhysicalPlan::Window {
            node: Box::new(WindowExecNode::new(window_exprs).with_cost(cost)),
            input: Box::new(input_plan),
        }
    }

    fn plan_join(&self, node: &JoinNode, left: &LogicalPlan, right: &LogicalPlan) -> PhysicalPlan {
        let left_plan = self.plan(left);
        let right_plan = self.plan(right);

        let left_rows = left_plan.cost().cardinality();
        let right_rows = right_plan.cost().cardinality();

        // Estimate output rows (rough heuristic)
        let output_rows = match node.join_type {
            JoinType::Cross => left_rows * right_rows,
            JoinType::Inner => (left_rows.min(right_rows) as f64 * 0.1).ceil() as usize,
            JoinType::Left => left_rows,
            JoinType::Right => right_rows,
            JoinType::Full => left_rows + right_rows,
            JoinType::LeftSemi | JoinType::LeftAnti => left_rows,
            JoinType::RightSemi | JoinType::RightAnti => right_rows,
        };

        // Handle USING columns by synthesizing equijoin conditions
        // For NATURAL JOIN / JOIN USING, using_columns contains the column names
        // that should be equal between left and right sides
        if !node.using_columns.is_empty() {
            // Build equijoin keys from USING columns
            // For USING(col1, col2), we need left.col1 = right.col1 AND left.col2 = right.col2
            let left_keys: Vec<LogicalExpr> =
                node.using_columns.iter().map(|col| LogicalExpr::column(col)).collect();
            let right_keys: Vec<LogicalExpr> =
                node.using_columns.iter().map(|col| LogicalExpr::column(col)).collect();

            // Use hash join for USING joins (they're always equijoins)
            let cost = self.cost_model.hash_join_cost(
                left_rows.min(right_rows),
                left_rows.max(right_rows),
                output_rows,
            );

            let (build_keys, probe_keys, join_order) = if left_rows <= right_rows {
                (left_keys, right_keys, JoinOrder::LeftBuild)
            } else {
                (right_keys, left_keys, JoinOrder::RightBuild)
            };

            let (build_plan, probe_plan) = match join_order {
                JoinOrder::LeftBuild => (left_plan, right_plan),
                JoinOrder::RightBuild => (right_plan, left_plan),
            };

            return PhysicalPlan::HashJoin {
                node: Box::new(
                    HashJoinNode::new(node.join_type, build_keys, probe_keys)
                        .with_join_order(join_order)
                        .with_cost(cost),
                ),
                build: Box::new(build_plan),
                probe: Box::new(probe_plan),
            };
        }

        // Check if we can use a hash join (need equijoin condition)
        if let Some(condition) = &node.condition {
            if let Some((left_key, right_key)) = self.extract_equijoin_keys(condition) {
                // Use hash join
                let cost = self.cost_model.hash_join_cost(
                    left_rows.min(right_rows),
                    left_rows.max(right_rows),
                    output_rows,
                );

                let (build_keys, probe_keys, join_order) = if left_rows <= right_rows {
                    (vec![left_key], vec![right_key], JoinOrder::LeftBuild)
                } else {
                    (vec![right_key], vec![left_key], JoinOrder::RightBuild)
                };

                let (build_plan, probe_plan) = match join_order {
                    JoinOrder::LeftBuild => (left_plan, right_plan),
                    JoinOrder::RightBuild => (right_plan, left_plan),
                };

                return PhysicalPlan::HashJoin {
                    node: Box::new(
                        HashJoinNode::new(node.join_type, build_keys, probe_keys)
                            .with_join_order(join_order)
                            .with_cost(cost),
                    ),
                    build: Box::new(build_plan),
                    probe: Box::new(probe_plan),
                };
            }
        }

        // Fall back to nested loop join
        // Note: USING joins are always handled by hash join above, so we use the original condition
        let cost = self.cost_model.nested_loop_cost(left_rows, right_rows, output_rows);

        PhysicalPlan::NestedLoopJoin {
            node: NestedLoopJoinNode::new(node.join_type, node.condition.clone()).with_cost(cost),
            left: Box::new(left_plan),
            right: Box::new(right_plan),
        }
    }

    fn plan_set_op(
        &self,
        node: &crate::plan::logical::SetOpNode,
        left: &LogicalPlan,
        right: &LogicalPlan,
    ) -> PhysicalPlan {
        let left_plan = self.plan(left);
        let right_plan = self.plan(right);

        let left_rows = left_plan.cost().cardinality();
        let right_rows = right_plan.cost().cardinality();
        let output_rows = left_rows + right_rows;

        let cost = Cost::new(output_rows as f64 * 0.1, output_rows);

        PhysicalPlan::SetOp {
            op_type: node.op_type,
            cost,
            left: Box::new(left_plan),
            right: Box::new(right_plan),
        }
    }

    fn plan_union(
        &self,
        node: &crate::plan::logical::UnionNode,
        inputs: &[LogicalPlan],
    ) -> PhysicalPlan {
        let input_plans: Vec<PhysicalPlan> = inputs.iter().map(|i| self.plan(i)).collect();

        let total_rows: usize = input_plans.iter().map(|p| p.cost().cardinality()).sum();
        let cost = Cost::new(total_rows as f64 * 0.1, total_rows);

        PhysicalPlan::Union { all: node.all, cost, inputs: input_plans }
    }

    fn plan_expand(&self, node: &ExpandNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_nodes = input_plan.cost().cardinality();

        // Estimate average degree (default 10)
        let avg_degree = 10.0;
        let cost = self.cost_model.graph_expand_cost(input_nodes, avg_degree);

        let mut exec_node = GraphExpandExecNode::new(&node.src_var, &node.dst_var, node.direction)
            .with_edge_types(node.edge_types.clone())
            .with_length(node.length.clone())
            .with_cost(cost);

        if let Some(var) = &node.edge_var {
            exec_node = exec_node.with_edge_var(var);
        }

        if let Some(filter) = &node.edge_filter {
            exec_node = exec_node.with_edge_filter(filter.clone());
        }

        if let Some(filter) = &node.node_filter {
            exec_node = exec_node.with_node_filter(filter.clone());
        }

        if !node.node_labels.is_empty() {
            exec_node = exec_node.with_node_labels(node.node_labels.clone());
        }

        PhysicalPlan::GraphExpand { node: Box::new(exec_node), input: Box::new(input_plan) }
    }

    fn plan_path_scan(&self, node: &PathScanNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let input_nodes = input_plan.cost().cardinality();

        // Estimate cost based on path length
        let avg_degree = 10.0;
        let total_cost =
            self.cost_model.graph_expand_cost(input_nodes * node.steps.len(), avg_degree);

        let steps: Vec<GraphExpandExecNode> = node
            .steps
            .iter()
            .map(|step| {
                GraphExpandExecNode::new(
                    &step.expand.src_var,
                    &step.expand.dst_var,
                    step.expand.direction,
                )
                .with_edge_types(step.expand.edge_types.clone())
                .with_length(step.expand.length.clone())
            })
            .collect();

        let mut exec_node = GraphPathScanExecNode::new(steps).with_cost(total_cost);

        if let Some(filter) = &node.start_filter {
            exec_node = exec_node.with_start_filter(filter.clone());
        }

        if node.all_paths {
            exec_node = exec_node.with_all_paths();
        }

        if node.track_path {
            exec_node = exec_node.with_track_path();
        }

        PhysicalPlan::GraphPathScan { node: Box::new(exec_node), input: Box::new(input_plan) }
    }

    fn plan_shortest_path(&self, node: &ShortestPathNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);

        // Estimate cost based on graph traversal
        // For BFS: O(V + E) where V is vertices and E is edges reachable within max_length
        // For all paths: multiply by expected number of paths
        let avg_degree: f64 = 10.0;
        let max_depth: f64 = node.max_length.unwrap_or(10) as f64;
        let base_cost = avg_degree.powf(max_depth.min(5.0)); // Cap exponential growth

        let cost = if node.find_all {
            // All shortest paths is more expensive - estimate multiple paths returned
            let estimated_paths = (base_cost * 2.0).ceil() as usize;
            Cost::new(base_cost * 2.0, estimated_paths)
        } else {
            // Single shortest path - returns one path
            Cost::new(base_cost, 1)
        };

        let mut exec_node = ShortestPathExecNode::new(&node.src_var, &node.dst_var)
            .with_direction(node.direction)
            .with_edge_types(node.edge_types.clone())
            .with_find_all(node.find_all)
            .with_src_labels(node.src_labels.clone())
            .with_dst_labels(node.dst_labels.clone())
            .with_cost(cost);

        if let Some(var) = &node.path_variable {
            exec_node = exec_node.with_path_variable(var);
        }

        if let Some(max) = node.max_length {
            exec_node = exec_node.with_max_length(max);
        }

        PhysicalPlan::ShortestPath { node: Box::new(exec_node), input: Box::new(input_plan) }
    }

    fn plan_ann_search(&self, node: &AnnSearchNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let table_rows = input_plan.cost().cardinality();

        // Get the source table/collection name
        let source_name = self.get_table_name(input);

        // Check for HNSW index - try named vector first, then fall back to table/column
        let hnsw_index_info = source_name.and_then(|table| {
            // First try to find an HNSW index for a named vector
            self.catalog.get_hnsw_index_for_named_vector(table, &node.vector_column).or_else(|| {
                // Fall back to regular table/column index lookup
                self.catalog.get_hnsw_index(table, &node.vector_column)
            })
        });

        // Choose algorithm based on data size and index availability
        let ef_search = node.params.ef_search.unwrap_or(node.k * 10);

        if let Some(index_info) = hnsw_index_info {
            if self.cost_model.prefer_hnsw(table_rows, node.k, ef_search) {
                // Use HNSW search with the index name
                let cost = self.cost_model.hnsw_search_cost(table_rows, node.k, ef_search);

                let mut hnsw_node = HnswSearchNode::new(
                    &node.vector_column,
                    node.query_vector.clone(),
                    node.metric,
                    node.k,
                )
                .with_ef_search(ef_search)
                .with_include_distance(node.include_distance)
                .with_index_name(&index_info.name)
                .with_cost(cost);

                // Set collection name if available
                if let Some(collection) = &index_info.collection_name {
                    hnsw_node = hnsw_node.with_collection_name(collection);
                }

                if let Some(filter) = &node.filter {
                    hnsw_node = hnsw_node.with_filter(filter.clone());
                }

                if let Some(alias) = &node.distance_alias {
                    hnsw_node = hnsw_node.with_distance_alias(alias);
                }

                return PhysicalPlan::HnswSearch {
                    node: Box::new(hnsw_node),
                    input: Box::new(input_plan),
                };
            }
        }

        // Fall back to brute-force search (no index or HNSW not preferred)
        let cost = self.cost_model.brute_force_search_cost(table_rows, node.k);

        let mut bf_node = BruteForceSearchNode::new(
            &node.vector_column,
            node.query_vector.clone(),
            node.metric,
            node.k,
        )
        .with_include_distance(node.include_distance)
        .with_cost(cost);

        if let Some(filter) = &node.filter {
            bf_node = bf_node.with_filter(filter.clone());
        }

        if let Some(alias) = &node.distance_alias {
            bf_node = bf_node.with_distance_alias(alias);
        }

        PhysicalPlan::BruteForceSearch { node: Box::new(bf_node), input: Box::new(input_plan) }
    }

    fn plan_vector_distance(
        &self,
        node: &crate::plan::logical::VectorDistanceNode,
        input: &LogicalPlan,
    ) -> PhysicalPlan {
        // Vector distance is just a projection with distance computation
        let input_plan = self.plan(input);
        let row_count = input_plan.cost().cardinality();
        let cost = Cost::new(row_count as f64 * self.cost_model.vector_distance_cost, row_count);

        // Map metric to the appropriate binary operator
        // Note: Manhattan and Hamming don't have direct SQL operators, fall back to Euclidean
        let op = match node.metric {
            crate::ast::DistanceMetric::Euclidean
            | crate::ast::DistanceMetric::Manhattan
            | crate::ast::DistanceMetric::Hamming => crate::ast::BinaryOp::EuclideanDistance,
            crate::ast::DistanceMetric::Cosine => crate::ast::BinaryOp::CosineDistance,
            crate::ast::DistanceMetric::InnerProduct => crate::ast::BinaryOp::InnerProduct,
        };

        let distance_expr = LogicalExpr::BinaryOp {
            left: Box::new(node.left.clone()),
            op,
            right: Box::new(node.right.clone()),
        };

        let expr =
            if let Some(alias) = &node.alias { distance_expr.alias(alias) } else { distance_expr };

        PhysicalPlan::Project {
            node: ProjectExecNode::new(vec![expr]).with_cost(cost),
            input: Box::new(input_plan),
        }
    }

    fn plan_hybrid_search(&self, node: &HybridSearchNode, input: &LogicalPlan) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let table_rows = input_plan.cost().cardinality();

        // Get the source table/collection name
        let source_name = self.get_table_name(input);

        // Build physical components for each search
        let mut components = Vec::with_capacity(node.components.len());
        let mut total_cost = 0.0;

        for comp in &node.components {
            // Check for HNSW index for this component
            let hnsw_index_info = source_name.and_then(|table| {
                self.catalog
                    .get_hnsw_index_for_named_vector(table, &comp.vector_column)
                    .or_else(|| self.catalog.get_hnsw_index(table, &comp.vector_column))
            });

            let ef_search = comp.params.ef_search.unwrap_or(node.k * 10);

            // Determine whether to use HNSW for this component
            let (use_hnsw, index_name, component_cost) = if let Some(index_info) = hnsw_index_info {
                if self.cost_model.prefer_hnsw(table_rows, node.k, ef_search) {
                    let cost = self.cost_model.hnsw_search_cost(table_rows, node.k, ef_search);
                    (true, Some(index_info.name.clone()), cost)
                } else {
                    let cost = self.cost_model.brute_force_search_cost(table_rows, node.k);
                    (false, None, cost)
                }
            } else {
                let cost = self.cost_model.brute_force_search_cost(table_rows, node.k);
                (false, None, cost)
            };

            total_cost += component_cost.value();

            let mut phys_comp = HybridSearchComponentNode::new(
                &comp.vector_column,
                comp.query_vector.clone(),
                comp.metric,
                comp.weight,
            )
            .with_ef_search(ef_search)
            .with_use_hnsw(use_hnsw);

            if let Some(name) = index_name {
                phys_comp = phys_comp.with_index_name(name);
            }

            components.push(phys_comp);
        }

        // Add merge overhead cost
        let merge_cost = (node.k * node.components.len()) as f64 * 0.1;
        total_cost += merge_cost;

        let cost = Cost::new(total_cost, node.k);

        // Convert combination method
        let combination_method = match node.combination_method {
            ScoreCombinationMethod::WeightedSum => PhysicalScoreCombinationMethod::WeightedSum,
            ScoreCombinationMethod::ReciprocalRankFusion { k_param } => {
                PhysicalScoreCombinationMethod::ReciprocalRankFusion { k_param }
            }
        };

        let mut phys_node = PhysicalHybridSearchNode::new(components, node.k)
            .with_combination_method(combination_method)
            .with_cost(cost);

        if !node.normalize_scores {
            phys_node = phys_node.without_normalization();
        }

        if let Some(filter) = &node.filter {
            phys_node = phys_node.with_filter(filter.clone());
        }

        if !node.include_score {
            phys_node = phys_node.with_include_score(false);
        }

        if let Some(alias) = &node.score_alias {
            phys_node = phys_node.with_score_alias(alias);
        }

        if let Some(source) = source_name {
            phys_node = phys_node.with_collection_name(source);
        }

        PhysicalPlan::HybridSearch { node: Box::new(phys_node), input: Box::new(input_plan) }
    }

    fn plan_insert(
        &self,
        table: &str,
        columns: &[String],
        input: &LogicalPlan,
        on_conflict: &Option<crate::plan::logical::LogicalOnConflict>,
        returning: &[LogicalExpr],
    ) -> PhysicalPlan {
        let input_plan = self.plan(input);
        let row_count = input_plan.cost().cardinality();
        let cost = Cost::new(row_count as f64 * 2.0, row_count);

        PhysicalPlan::Insert {
            table: table.to_string(),
            columns: columns.to_vec(),
            on_conflict: on_conflict.clone(),
            returning: returning.to_vec(),
            cost,
            input: Box::new(input_plan),
        }
    }

    fn plan_update(
        &self,
        table: &str,
        assignments: &[(String, LogicalExpr)],
        filter: &Option<LogicalExpr>,
        returning: &[LogicalExpr],
    ) -> PhysicalPlan {
        let row_count = self.catalog.get_row_count(table);
        let affected =
            if filter.is_some() { (row_count as f64 * 0.1).ceil() as usize } else { row_count };
        let cost = Cost::new(affected as f64 * 3.0, affected);

        PhysicalPlan::Update {
            table: table.to_string(),
            assignments: assignments.to_vec(),
            filter: filter.clone(),
            returning: returning.to_vec(),
            cost,
        }
    }

    fn plan_delete(
        &self,
        table: &str,
        filter: &Option<LogicalExpr>,
        returning: &[LogicalExpr],
    ) -> PhysicalPlan {
        let row_count = self.catalog.get_row_count(table);
        let affected =
            if filter.is_some() { (row_count as f64 * 0.1).ceil() as usize } else { row_count };
        let cost = Cost::new(affected as f64 * 2.0, affected);

        PhysicalPlan::Delete {
            table: table.to_string(),
            filter: filter.clone(),
            returning: returning.to_vec(),
            cost,
        }
    }

    fn plan_merge_sql(
        &self,
        target_table: &str,
        source: &LogicalPlan,
        on_condition: &LogicalExpr,
        clauses: &[crate::plan::logical::LogicalMergeClause],
    ) -> PhysicalPlan {
        let source_plan = self.plan(source);
        let source_rows = source_plan.cost().cardinality();
        let target_rows = self.catalog.get_row_count(target_table);

        // MERGE cost: source scan + join + actions for matched/unmatched rows
        // Rough estimate: source rows * 3 (for comparison and potential actions)
        let estimated_affected = source_rows.max(target_rows);
        let cost = Cost::new(estimated_affected as f64 * 3.0, estimated_affected);

        PhysicalPlan::MergeSql {
            target_table: target_table.to_string(),
            source: Box::new(source_plan),
            on_condition: on_condition.clone(),
            clauses: clauses.to_vec(),
            cost,
        }
    }

    // ========== Helper Methods ==========

    /// Estimates selectivity of a predicate (0.0 to 1.0).
    fn estimate_selectivity(&self, _predicate: &LogicalExpr) -> f64 {
        // Simple heuristic; real implementation would analyze the predicate
        0.5
    }

    /// Extracts equijoin keys from a condition if possible.
    fn extract_equijoin_keys(&self, condition: &LogicalExpr) -> Option<(LogicalExpr, LogicalExpr)> {
        if let LogicalExpr::BinaryOp { left, op, right } = condition {
            if matches!(op, crate::ast::BinaryOp::Eq) {
                // Check if left and right are columns from different sides
                if matches!(left.as_ref(), LogicalExpr::Column { .. })
                    && matches!(right.as_ref(), LogicalExpr::Column { .. })
                {
                    return Some((*left.clone(), *right.clone()));
                }
            }
        }
        None
    }

    /// Gets the table name from a logical plan if it's a scan.
    fn get_table_name<'a>(&self, plan: &'a LogicalPlan) -> Option<&'a str> {
        match plan {
            LogicalPlan::Scan(node) => Some(&node.table_name),
            LogicalPlan::Alias { input, .. } => self.get_table_name(input),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::{LogicalExpr, SortOrder};

    #[test]
    fn plan_simple_scan() {
        let logical = LogicalPlan::Scan(Box::new(ScanNode::new("users")));
        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "FullScan");
    }

    #[test]
    fn plan_filter_on_scan() {
        let logical = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "Filter");
        assert_eq!(physical.children().len(), 1);
        assert_eq!(physical.children()[0].node_type(), "FullScan");
    }

    #[test]
    fn plan_projection() {
        let logical = LogicalPlan::scan("users")
            .project(vec![LogicalExpr::column("id"), LogicalExpr::column("name")]);

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "Project");
    }

    #[test]
    fn plan_sort() {
        let logical =
            LogicalPlan::scan("users").sort(vec![SortOrder::asc(LogicalExpr::column("name"))]);

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "Sort");
    }

    #[test]
    fn plan_aggregate() {
        let logical = LogicalPlan::scan("orders").aggregate(
            vec![LogicalExpr::column("status")],
            vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
        );

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashAggregate");
    }

    #[test]
    fn plan_nested_loop_join() {
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        // Non-equijoin condition triggers nested loop
        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::inner(
                LogicalExpr::column("users.id").gt(LogicalExpr::column("orders.user_id")),
            )),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "NestedLoopJoin");
    }

    #[test]
    fn plan_hash_join() {
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        // Equijoin condition triggers hash join
        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::inner(
                LogicalExpr::column("users.id").eq(LogicalExpr::column("orders.user_id")),
            )),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashJoin");
    }

    #[test]
    fn plan_distinct() {
        let logical = LogicalPlan::scan("users").distinct();

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashDistinct");
    }

    #[test]
    fn plan_limit() {
        let logical = LogicalPlan::scan("users").limit(10);

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "Limit");
    }

    #[test]
    fn plan_with_catalog() {
        let catalog = PlannerCatalog::new()
            .with_table(TableStats::new("users", 1_000_000))
            .with_index(IndexInfo::btree("users_age_idx", "users", vec!["age".to_string()]));

        let planner = PhysicalPlanner::new().with_catalog(catalog);
        let logical = LogicalPlan::scan("users");
        let physical = planner.plan(&logical);

        // Should use full scan (no predicate to match index)
        assert_eq!(physical.node_type(), "FullScan");
    }

    #[test]
    fn plan_complex_query() {
        // SELECT u.name, COUNT(*) as order_count
        // FROM users u
        // JOIN orders o ON u.id = o.user_id
        // WHERE u.active = true
        // GROUP BY u.name
        // ORDER BY order_count DESC
        // LIMIT 10

        let users = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("active").eq(LogicalExpr::boolean(true)));

        let orders = LogicalPlan::scan("orders");

        let joined = LogicalPlan::Join {
            node: Box::new(JoinNode::inner(
                LogicalExpr::column("id").eq(LogicalExpr::column("user_id")),
            )),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let aggregated = joined.aggregate(
            vec![LogicalExpr::column("name")],
            vec![LogicalExpr::count(LogicalExpr::wildcard(), false).alias("order_count")],
        );

        let sorted = aggregated.sort(vec![SortOrder::desc(LogicalExpr::column("order_count"))]);

        let limited = sorted.limit(10);

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&limited);

        // Verify the plan structure
        assert_eq!(physical.node_type(), "Limit");

        let sort = &physical.children()[0];
        assert_eq!(sort.node_type(), "Sort");

        let agg = &sort.children()[0];
        assert_eq!(agg.node_type(), "HashAggregate");

        let join = &agg.children()[0];
        assert_eq!(join.node_type(), "HashJoin");
    }

    #[test]
    fn display_physical_plan() {
        let logical = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)))
            .project(vec![LogicalExpr::column("id"), LogicalExpr::column("name")])
            .limit(10);

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        let display = format!("{physical}");
        assert!(display.contains("Limit"));
        assert!(display.contains("Project"));
        assert!(display.contains("Filter"));
        assert!(display.contains("FullScan"));
        assert!(display.contains("cost:"));
    }

    #[test]
    fn hnsw_index_for_named_vector() {
        // Test that HNSW index info correctly stores collection and vector metadata
        let index = IndexInfo::hnsw_for_named_vector("documents", "embedding");

        assert_eq!(index.name, "documents_embedding_hnsw");
        assert_eq!(index.table, "documents");
        assert_eq!(index.columns, vec!["embedding".to_string()]);
        assert_eq!(index.collection_name, Some("documents".to_string()));
        assert_eq!(index.vector_name, Some("embedding".to_string()));
        assert_eq!(index.index_type, IndexType::Hnsw);
    }

    #[test]
    fn catalog_lookup_named_vector_index() {
        // Test catalog can find HNSW indexes by collection and vector name
        let catalog = PlannerCatalog::new()
            .with_index(IndexInfo::hnsw_for_named_vector("documents", "dense"))
            .with_index(IndexInfo::hnsw_for_named_vector("documents", "sparse"))
            .with_index(IndexInfo::hnsw_for_named_vector("images", "embedding"));

        // Should find the correct index for documents/dense
        let dense_index = catalog.get_hnsw_index_for_named_vector("documents", "dense");
        assert!(dense_index.is_some());
        assert_eq!(dense_index.unwrap().name, "documents_dense_hnsw");

        // Should find the correct index for documents/sparse
        let sparse_index = catalog.get_hnsw_index_for_named_vector("documents", "sparse");
        assert!(sparse_index.is_some());
        assert_eq!(sparse_index.unwrap().name, "documents_sparse_hnsw");

        // Should find the correct index for images/embedding
        let images_index = catalog.get_hnsw_index_for_named_vector("images", "embedding");
        assert!(images_index.is_some());
        assert_eq!(images_index.unwrap().name, "images_embedding_hnsw");

        // Should not find non-existent indexes
        assert!(catalog.get_hnsw_index_for_named_vector("documents", "nonexistent").is_none());
        assert!(catalog.get_hnsw_index_for_named_vector("nonexistent", "dense").is_none());
    }

    #[test]
    fn catalog_has_named_vector_index() {
        let catalog =
            PlannerCatalog::new().with_index(IndexInfo::hnsw_for_named_vector("docs", "text"));

        assert!(catalog.has_hnsw_index_for_named_vector("docs", "text"));
        assert!(!catalog.has_hnsw_index_for_named_vector("docs", "other"));
        assert!(!catalog.has_hnsw_index_for_named_vector("other", "text"));
    }

    #[test]
    fn plan_ann_search_with_named_vector_index() {
        use crate::plan::logical::AnnSearchNode;

        // Set up catalog with a named vector index
        let catalog = PlannerCatalog::new()
            .with_table(TableStats::new("documents", 100_000))
            .with_index(IndexInfo::hnsw_for_named_vector("documents", "embedding"));

        let planner = PhysicalPlanner::new().with_catalog(catalog);

        // Create an ANN search on the documents.embedding vector
        let scan = LogicalPlan::scan("documents");
        let ann_node = AnnSearchNode::cosine("embedding", LogicalExpr::param(1), 10);
        let logical = LogicalPlan::AnnSearch { node: Box::new(ann_node), input: Box::new(scan) };

        let physical = planner.plan(&logical);

        // Should use HNSW search since we have an index
        assert_eq!(physical.node_type(), "HnswSearch");

        // Verify the index name is populated
        if let PhysicalPlan::HnswSearch { node, .. } = &physical {
            assert_eq!(node.index_name, Some("documents_embedding_hnsw".to_string()));
            assert_eq!(node.collection_name, Some("documents".to_string()));
        } else {
            panic!("Expected HnswSearch plan");
        }
    }

    #[test]
    fn plan_ann_search_without_index_falls_back_to_brute_force() {
        use crate::plan::logical::AnnSearchNode;

        // No indexes in catalog
        let catalog = PlannerCatalog::new().with_table(TableStats::new("documents", 1000));

        let planner = PhysicalPlanner::new().with_catalog(catalog);

        // Create an ANN search
        let scan = LogicalPlan::scan("documents");
        let ann_node = AnnSearchNode::cosine("embedding", LogicalExpr::param(1), 10);
        let logical = LogicalPlan::AnnSearch { node: Box::new(ann_node), input: Box::new(scan) };

        let physical = planner.plan(&logical);

        // Should fall back to brute force since no index
        assert_eq!(physical.node_type(), "BruteForceSearch");
    }

    #[test]
    fn hnsw_search_node_for_named_vector() {
        use super::HnswSearchNode;
        use crate::ast::DistanceMetric;

        // Test the for_named_vector constructor
        let node = HnswSearchNode::for_named_vector(
            "documents",
            "embedding",
            LogicalExpr::param(1),
            DistanceMetric::Cosine,
            10,
        );

        assert_eq!(node.vector_column, "embedding");
        assert_eq!(node.index_name, Some("documents_embedding_hnsw".to_string()));
        assert_eq!(node.collection_name, Some("documents".to_string()));
        assert_eq!(node.k, 10);
    }

    #[test]
    fn hnsw_search_node_with_collection() {
        use super::HnswSearchNode;
        use crate::ast::DistanceMetric;

        // Test building with with_collection
        let node =
            HnswSearchNode::new("dense", LogicalExpr::param(1), DistanceMetric::Euclidean, 5)
                .with_collection("my_collection");

        assert_eq!(node.vector_column, "dense");
        assert_eq!(node.index_name, Some("my_collection_dense_hnsw".to_string()));
        assert_eq!(node.collection_name, Some("my_collection".to_string()));
    }

    #[test]
    fn plan_join_using_single_column() {
        // Test JOIN ... USING(id) creates a hash join with proper keys
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::using(JoinType::Inner, vec!["id".to_string()])),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        // USING joins should use hash join since they're equijoins
        assert_eq!(physical.node_type(), "HashJoin");

        // Verify hash join has proper keys
        if let PhysicalPlan::HashJoin { node, .. } = &physical {
            assert_eq!(node.build_keys.len(), 1);
            assert_eq!(node.probe_keys.len(), 1);
            // Keys should be column references to "id"
            assert!(
                matches!(&node.build_keys[0], LogicalExpr::Column { name, .. } if name == "id")
            );
            assert!(
                matches!(&node.probe_keys[0], LogicalExpr::Column { name, .. } if name == "id")
            );
        } else {
            panic!("Expected HashJoin plan");
        }
    }

    #[test]
    fn plan_join_using_multiple_columns() {
        // Test JOIN ... USING(dept_id, location_id) creates hash join with multiple keys
        let employees = LogicalPlan::scan("employees");
        let departments = LogicalPlan::scan("departments");

        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::using(
                JoinType::Inner,
                vec!["dept_id".to_string(), "location_id".to_string()],
            )),
            left: Box::new(employees),
            right: Box::new(departments),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashJoin");

        if let PhysicalPlan::HashJoin { node, .. } = &physical {
            // Should have two keys for the two USING columns
            assert_eq!(node.build_keys.len(), 2);
            assert_eq!(node.probe_keys.len(), 2);
        } else {
            panic!("Expected HashJoin plan");
        }
    }

    #[test]
    fn plan_natural_join() {
        // NATURAL JOIN is represented as a JOIN with using_columns populated
        // by the logical planner (common columns between tables)
        let users = LogicalPlan::scan("users");
        let profiles = LogicalPlan::scan("profiles");

        // Simulate NATURAL JOIN - assuming logical planner identified "user_id" as common
        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::using(JoinType::Inner, vec!["user_id".to_string()])),
            left: Box::new(users),
            right: Box::new(profiles),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        // NATURAL JOIN should use hash join
        assert_eq!(physical.node_type(), "HashJoin");
    }

    #[test]
    fn plan_left_join_using() {
        // Test LEFT JOIN ... USING(id)
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::using(JoinType::Left, vec!["id".to_string()])),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashJoin");

        if let PhysicalPlan::HashJoin { node, .. } = &physical {
            assert_eq!(node.join_type, JoinType::Left);
        } else {
            panic!("Expected HashJoin plan");
        }
    }

    #[test]
    fn plan_right_join_using() {
        // Test RIGHT JOIN ... USING(id)
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::using(JoinType::Right, vec!["id".to_string()])),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashJoin");

        if let PhysicalPlan::HashJoin { node, .. } = &physical {
            assert_eq!(node.join_type, JoinType::Right);
        } else {
            panic!("Expected HashJoin plan");
        }
    }

    #[test]
    fn plan_full_join_using() {
        // Test FULL JOIN ... USING(id)
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        let logical = LogicalPlan::Join {
            node: Box::new(JoinNode::using(JoinType::Full, vec!["id".to_string()])),
            left: Box::new(users),
            right: Box::new(orders),
        };

        let planner = PhysicalPlanner::new();
        let physical = planner.plan(&logical);

        assert_eq!(physical.node_type(), "HashJoin");

        if let PhysicalPlan::HashJoin { node, .. } = &physical {
            assert_eq!(node.join_type, JoinType::Full);
        } else {
            panic!("Expected HashJoin plan");
        }
    }
}
