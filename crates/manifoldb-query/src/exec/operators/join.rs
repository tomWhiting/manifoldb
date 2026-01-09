//! Join operators for combining data from multiple sources.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::Value;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{JoinType, LogicalExpr};

/// Nested loop join operator.
///
/// Simple O(n*m) join that evaluates a condition for each pair of rows.
pub struct NestedLoopJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Join condition.
    condition: Option<LogicalExpr>,
    /// Left (outer) input.
    left: BoxedOperator,
    /// Right (inner) input.
    right: BoxedOperator,
    /// Materialized right rows.
    right_rows: Vec<Row>,
    /// Current left row.
    current_left: Option<Row>,
    /// Current position in right rows.
    right_position: usize,
    /// Whether we've matched current left row.
    matched_left: bool,
    /// Whether right is materialized.
    right_materialized: bool,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl NestedLoopJoinOp {
    /// Creates a new nested loop join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        condition: Option<LogicalExpr>,
        left: BoxedOperator,
        right: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(left.schema().merge(&right.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            condition,
            left,
            right,
            right_rows: Vec::new(),
            current_left: None,
            right_position: 0,
            matched_left: false,
            right_materialized: false,
            max_rows_in_memory: 0, // Set in open() from context
        }
    }

    /// Evaluates the join condition.
    fn matches(&self, left: &Row, right: &Row) -> OperatorResult<bool> {
        match &self.condition {
            Some(cond) => {
                // Merge rows to evaluate condition
                let merged = left.merge(right);
                let result = evaluate_expr(cond, &merged)?;
                match result {
                    Value::Bool(b) => Ok(b),
                    Value::Null => Ok(false),
                    _ => Ok(false),
                }
            }
            None => Ok(true), // Cross join
        }
    }

    /// Creates a null row for the right side.
    fn right_null_row(&self) -> Row {
        Row::empty(self.right.schema())
    }

    /// Creates a null row for the left side.
    ///
    /// Used for RIGHT and FULL outer joins to output right rows
    /// that have no matching left rows.
    pub fn left_null_row(&self) -> Row {
        Row::empty(self.left.schema())
    }

    /// Returns the join type.
    #[must_use]
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }
}

impl Operator for NestedLoopJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.left.open(ctx)?;
        self.right.open(ctx)?;
        self.right_rows.clear();
        self.current_left = None;
        self.right_position = 0;
        self.matched_left = false;
        self.right_materialized = false;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Materialize right side on first call with size limit check
        if !self.right_materialized {
            while let Some(row) = self.right.next()? {
                self.right_rows.push(row);

                // Check limit after each row (0 means no limit)
                if self.max_rows_in_memory > 0 && self.right_rows.len() > self.max_rows_in_memory {
                    return Err(ParseError::QueryTooLarge {
                        actual: self.right_rows.len(),
                        limit: self.max_rows_in_memory,
                    });
                }
            }
            self.right_materialized = true;
        }

        loop {
            // Get next left row if needed
            if self.current_left.is_none() {
                match self.left.next()? {
                    Some(row) => {
                        self.current_left = Some(row);
                        self.right_position = 0;
                        self.matched_left = false;
                    }
                    None => {
                        self.base.set_finished();
                        return Ok(None);
                    }
                }
            }

            // Process current left row - use if let to avoid unwrap
            // State invariant: current_left is Some after the block above
            if let Some(left_row) = &self.current_left {
                // Try to find matching right row
                while self.right_position < self.right_rows.len() {
                    let right_row = &self.right_rows[self.right_position];
                    self.right_position += 1;

                    if self.matches(left_row, right_row)? {
                        self.matched_left = true;
                        self.base.inc_rows_produced();
                        return Ok(Some(left_row.merge(right_row)));
                    }
                }

                // No more right rows for this left row
                match self.join_type {
                    JoinType::Left | JoinType::Full => {
                        if !self.matched_left {
                            // Output left row with NULL right
                            self.matched_left = true;
                            self.base.inc_rows_produced();
                            return Ok(Some(left_row.merge(&self.right_null_row())));
                        }
                    }
                    _ => {}
                }
            }

            // Move to next left row
            self.current_left = None;
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.left.close()?;
        self.right.close()?;
        self.right_rows.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "NestedLoopJoin"
    }
}

/// Hash join operator.
///
/// Builds a hash table on the build side, probes with the probe side.
/// Efficient for equijoin conditions.
pub struct HashJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Build side key expressions.
    build_keys: Vec<LogicalExpr>,
    /// Probe side key expressions.
    probe_keys: Vec<LogicalExpr>,
    /// Additional filter.
    filter: Option<LogicalExpr>,
    /// Build (left) input.
    build: BoxedOperator,
    /// Probe (right) input.
    probe: BoxedOperator,
    /// Hash table: key -> shared list of matching rows (Arc avoids cloning on probe).
    hash_table: HashMap<Vec<u8>, Arc<Vec<Row>>>,
    /// Current probe row.
    current_probe: Option<Row>,
    /// Current matches for probe row (shared reference, no clone).
    current_matches: Option<Arc<Vec<Row>>>,
    /// Position in current matches.
    match_position: usize,
    /// Whether hash table is built.
    built: bool,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl HashJoinOp {
    /// Creates a new hash join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        build_keys: Vec<LogicalExpr>,
        probe_keys: Vec<LogicalExpr>,
        filter: Option<LogicalExpr>,
        build: BoxedOperator,
        probe: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(build.schema().merge(&probe.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            build_keys,
            probe_keys,
            filter,
            build,
            probe,
            hash_table: HashMap::new(),
            current_probe: None,
            current_matches: None,
            match_position: 0,
            built: false,
            max_rows_in_memory: 0, // Set in open() from context
        }
    }

    /// Computes hash key from expressions.
    fn compute_key(&self, row: &Row, exprs: &[LogicalExpr]) -> OperatorResult<Vec<u8>> {
        // Pre-allocate for typical key sizes (64 bytes handles most cases)
        let mut key = Vec::with_capacity(64);
        for expr in exprs {
            let value = evaluate_expr(expr, row)?;
            // Simple key encoding
            match &value {
                Value::Null => key.push(0),
                Value::Bool(b) => {
                    key.push(1);
                    key.push(u8::from(*b));
                }
                Value::Int(i) => {
                    key.push(2);
                    key.extend_from_slice(&i.to_le_bytes());
                }
                Value::Float(f) => {
                    key.push(3);
                    key.extend_from_slice(&f.to_le_bytes());
                }
                Value::String(s) => {
                    key.push(4);
                    key.extend_from_slice(s.as_bytes());
                    key.push(0); // Null terminator
                }
                _ => key.push(0),
            }
        }
        Ok(key)
    }

    /// Builds the hash table from the build side.
    fn build_hash_table(&mut self) -> OperatorResult<()> {
        // First, collect all rows into a temporary HashMap with Vec
        // Pre-allocate for typical join sizes
        const INITIAL_CAPACITY: usize = 1000;
        let mut temp_table: HashMap<Vec<u8>, Vec<Row>> = HashMap::with_capacity(INITIAL_CAPACITY);
        let mut total_rows = 0usize;
        while let Some(row) = self.build.next()? {
            let key = self.compute_key(&row, &self.build_keys)?;
            temp_table.entry(key).or_default().push(row);
            total_rows += 1;

            // Check limit after each row (0 means no limit)
            if self.max_rows_in_memory > 0 && total_rows > self.max_rows_in_memory {
                return Err(ParseError::QueryTooLarge {
                    actual: total_rows,
                    limit: self.max_rows_in_memory,
                });
            }
        }
        // Convert to Arc<Vec<Row>> for sharing without cloning
        for (key, rows) in temp_table {
            self.hash_table.insert(key, Arc::new(rows));
        }
        self.built = true;
        Ok(())
    }

    /// Checks if filter passes.
    fn filter_passes(&self, left: &Row, right: &Row) -> OperatorResult<bool> {
        match &self.filter {
            Some(f) => {
                let merged = left.merge(right);
                let result = evaluate_expr(f, &merged)?;
                Ok(matches!(result, Value::Bool(true)))
            }
            None => Ok(true),
        }
    }

    /// Creates a null row for the build side.
    fn build_null_row(&self) -> Row {
        Row::empty(self.build.schema())
    }
}

impl Operator for HashJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.build.open(ctx)?;
        self.probe.open(ctx)?;
        self.hash_table.clear();
        self.current_probe = None;
        self.current_matches = None;
        self.match_position = 0;
        self.built = false;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Build hash table on first call
        if !self.built {
            self.build_hash_table()?;
        }

        loop {
            // Return next match if available
            if let Some(ref matches) = self.current_matches {
                while self.match_position < matches.len() {
                    let build_row = &matches[self.match_position];
                    self.match_position += 1;

                    if let Some(probe_row) = &self.current_probe {
                        if self.filter_passes(build_row, probe_row)? {
                            self.base.inc_rows_produced();
                            return Ok(Some(build_row.merge(probe_row)));
                        }
                    }
                }
            }

            // Get next probe row
            match self.probe.next()? {
                Some(probe_row) => {
                    let key = self.compute_key(&probe_row, &self.probe_keys)?;
                    // Clone only the Arc pointer, not the underlying Vec<Row>
                    self.current_matches = self.hash_table.get(&key).cloned();
                    self.match_position = 0;

                    // Handle left outer join with no matches
                    let has_no_matches =
                        self.current_matches.as_ref().map_or(true, |m| m.is_empty());
                    if has_no_matches && self.join_type == JoinType::Left {
                        self.base.inc_rows_produced();
                        // Use probe_row directly before moving it to current_probe
                        let result = self.build_null_row().merge(&probe_row);
                        self.current_probe = Some(probe_row);
                        return Ok(Some(result));
                    }
                    self.current_probe = Some(probe_row);
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.build.close()?;
        self.probe.close()?;
        self.hash_table.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "HashJoin"
    }
}

/// Merge join operator.
///
/// Merges two sorted inputs. Requires both sides sorted on join keys.
pub struct MergeJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Left side key expressions.
    left_keys: Vec<LogicalExpr>,
    /// Right side key expressions.
    right_keys: Vec<LogicalExpr>,
    /// Left input.
    left: BoxedOperator,
    /// Right input.
    right: BoxedOperator,
    /// Current left row.
    current_left: Option<Row>,
    /// Current right row.
    current_right: Option<Row>,
    /// Buffer of right rows with same key.
    right_buffer: Vec<Row>,
    /// Position in right buffer.
    buffer_position: usize,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl MergeJoinOp {
    /// Creates a new merge join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        left_keys: Vec<LogicalExpr>,
        right_keys: Vec<LogicalExpr>,
        left: BoxedOperator,
        right: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(left.schema().merge(&right.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            left_keys,
            right_keys,
            left,
            right,
            current_left: None,
            current_right: None,
            right_buffer: Vec::new(),
            buffer_position: 0,
            max_rows_in_memory: 0, // Set in open() from context
        }
    }

    /// Returns the join type.
    #[must_use]
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Compares keys from two rows.
    fn compare_keys(&self, left: &Row, right: &Row) -> OperatorResult<std::cmp::Ordering> {
        for (left_expr, right_expr) in self.left_keys.iter().zip(&self.right_keys) {
            let left_val = evaluate_expr(left_expr, left)?;
            let right_val = evaluate_expr(right_expr, right)?;

            let cmp = compare_values(&left_val, &right_val);
            if cmp != std::cmp::Ordering::Equal {
                return Ok(cmp);
            }
        }
        Ok(std::cmp::Ordering::Equal)
    }
}

impl Operator for MergeJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.left.open(ctx)?;
        self.right.open(ctx)?;
        self.current_left = self.left.next()?;
        self.current_right = self.right.next()?;
        self.right_buffer.clear();
        self.buffer_position = 0;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // Return from buffer if available
            if self.buffer_position < self.right_buffer.len() {
                if let Some(left) = &self.current_left {
                    let right = &self.right_buffer[self.buffer_position];
                    self.buffer_position += 1;
                    self.base.inc_rows_produced();
                    return Ok(Some(left.merge(right)));
                }
            }

            // Need both sides to continue
            let (left, right) = match (&self.current_left, &self.current_right) {
                (Some(l), Some(r)) => (l, r),
                _ => {
                    self.base.set_finished();
                    return Ok(None);
                }
            };

            match self.compare_keys(left, right)? {
                std::cmp::Ordering::Less => {
                    // Advance left
                    self.current_left = self.left.next()?;
                    self.right_buffer.clear();
                    self.buffer_position = 0;
                }
                std::cmp::Ordering::Greater => {
                    // Advance right
                    self.current_right = self.right.next()?;
                }
                std::cmp::Ordering::Equal => {
                    // Buffer all right rows with same key
                    self.right_buffer.clear();
                    self.right_buffer.push(right.clone());

                    // Read more right rows with same key
                    loop {
                        self.current_right = self.right.next()?;
                        match &self.current_right {
                            Some(r) if self.compare_keys(left, r)? == std::cmp::Ordering::Equal => {
                                self.right_buffer.push(r.clone());

                                // Check limit after each row (0 means no limit)
                                if self.max_rows_in_memory > 0
                                    && self.right_buffer.len() > self.max_rows_in_memory
                                {
                                    return Err(ParseError::QueryTooLarge {
                                        actual: self.right_buffer.len(),
                                        limit: self.max_rows_in_memory,
                                    });
                                }
                            }
                            _ => break,
                        }
                    }

                    self.buffer_position = 0;
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.left.close()?;
        self.right.close()?;
        self.right_buffer.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "MergeJoin"
    }
}

/// Compares two values.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Index nested loop join operator.
///
/// Accelerates nested loop joins by using an index on the inner (right) table
/// for key lookups instead of full scans. This is efficient when:
/// - The inner table has an index on the join key
/// - The outer table is relatively small
/// - The join is on an equality condition
///
/// Time complexity: O(n * log(m)) where n is outer rows, m is inner rows
/// (assuming B-tree index with O(log m) lookup)
pub struct IndexNestedLoopJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Outer (left) table key expression.
    outer_key: LogicalExpr,
    /// Inner (right) table key expression.
    inner_key: LogicalExpr,
    /// Additional filter condition (beyond the key equality).
    filter: Option<LogicalExpr>,
    /// Outer (left) input operator.
    outer: BoxedOperator,
    /// Inner (right) input operator - represents the indexed table.
    inner: BoxedOperator,
    /// Index name being used.
    index_name: String,
    /// Hash table for index simulation: key -> list of rows.
    /// In a real implementation, this would use actual index lookups.
    index_data: HashMap<Vec<u8>, Vec<Row>>,
    /// Current outer row being processed.
    current_outer: Option<Row>,
    /// Current matches from index lookup.
    current_matches: Vec<Row>,
    /// Position in current matches.
    match_position: usize,
    /// Whether the inner table has been indexed.
    indexed: bool,
    /// Whether current outer row has been matched.
    matched_outer: bool,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl IndexNestedLoopJoinOp {
    /// Creates a new index nested loop join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        outer_key: LogicalExpr,
        inner_key: LogicalExpr,
        index_name: impl Into<String>,
        outer: BoxedOperator,
        inner: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(outer.schema().merge(&inner.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            outer_key,
            inner_key,
            filter: None,
            outer,
            inner,
            index_name: index_name.into(),
            index_data: HashMap::new(),
            current_outer: None,
            current_matches: Vec::new(),
            match_position: 0,
            indexed: false,
            matched_outer: false,
            max_rows_in_memory: 0,
        }
    }

    /// Sets an additional filter condition.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Returns the index name being used.
    #[must_use]
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Computes the key value for index lookup.
    fn compute_key(&self, row: &Row, expr: &LogicalExpr) -> OperatorResult<Vec<u8>> {
        let value = evaluate_expr(expr, row)?;
        let mut key = Vec::with_capacity(64);
        encode_key_value(&value, &mut key);
        Ok(key)
    }

    /// Builds the index from inner table rows.
    fn build_index(&mut self) -> OperatorResult<()> {
        let mut total_rows = 0usize;
        while let Some(row) = self.inner.next()? {
            let key = self.compute_key(&row, &self.inner_key)?;
            self.index_data.entry(key).or_default().push(row);
            total_rows += 1;

            if self.max_rows_in_memory > 0 && total_rows > self.max_rows_in_memory {
                return Err(ParseError::QueryTooLarge {
                    actual: total_rows,
                    limit: self.max_rows_in_memory,
                });
            }
        }
        self.indexed = true;
        Ok(())
    }

    /// Looks up rows from the index matching the given key.
    fn index_lookup(&self, key: &[u8]) -> Vec<Row> {
        self.index_data.get(key).cloned().unwrap_or_default()
    }

    /// Checks if filter passes.
    fn filter_passes(&self, outer: &Row, inner: &Row) -> OperatorResult<bool> {
        match &self.filter {
            Some(f) => {
                let merged = outer.merge(inner);
                let result = evaluate_expr(f, &merged)?;
                Ok(matches!(result, Value::Bool(true)))
            }
            None => Ok(true),
        }
    }

    /// Creates a null row for the inner side.
    fn inner_null_row(&self) -> Row {
        Row::empty(self.inner.schema())
    }
}

impl Operator for IndexNestedLoopJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.outer.open(ctx)?;
        self.inner.open(ctx)?;
        self.index_data.clear();
        self.current_outer = None;
        self.current_matches.clear();
        self.match_position = 0;
        self.indexed = false;
        self.matched_outer = false;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Build index on first call
        if !self.indexed {
            self.build_index()?;
        }

        loop {
            // Return next match if available
            while self.match_position < self.current_matches.len() {
                let inner_row = &self.current_matches[self.match_position];
                self.match_position += 1;

                if let Some(outer_row) = &self.current_outer {
                    if self.filter_passes(outer_row, inner_row)? {
                        self.matched_outer = true;
                        self.base.inc_rows_produced();
                        return Ok(Some(outer_row.merge(inner_row)));
                    }
                }
            }

            // Handle unmatched outer row for left/full outer joins
            if let Some(outer_row) = &self.current_outer {
                if !self.matched_outer {
                    match self.join_type {
                        JoinType::Left | JoinType::Full => {
                            self.matched_outer = true;
                            self.base.inc_rows_produced();
                            return Ok(Some(outer_row.merge(&self.inner_null_row())));
                        }
                        _ => {}
                    }
                }
            }

            // Get next outer row
            match self.outer.next()? {
                Some(outer_row) => {
                    // Perform index lookup
                    let key = self.compute_key(&outer_row, &self.outer_key)?;
                    self.current_matches = self.index_lookup(&key);
                    self.match_position = 0;
                    self.matched_outer = false;
                    self.current_outer = Some(outer_row);
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.outer.close()?;
        self.inner.close()?;
        self.index_data.clear();
        self.current_matches.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "IndexNestedLoopJoin"
    }
}

/// Sort-merge join operator.
///
/// Efficient join algorithm for pre-sorted or sortable inputs.
/// Merges two sorted streams by advancing through both in sorted order.
///
/// Supports INNER, LEFT, RIGHT, and FULL outer joins.
///
/// Time complexity: O(n + m) for sorted inputs where n, m are input sizes.
/// Space complexity: O(k) where k is the maximum number of rows with the same key.
pub struct SortMergeJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Left side key expressions.
    left_keys: Vec<LogicalExpr>,
    /// Right side key expressions.
    right_keys: Vec<LogicalExpr>,
    /// Additional filter condition.
    filter: Option<LogicalExpr>,
    /// Left input (must be sorted on left_keys).
    left: BoxedOperator,
    /// Right input (must be sorted on right_keys).
    right: BoxedOperator,
    /// Current left row.
    current_left: Option<Row>,
    /// Current right row.
    current_right: Option<Row>,
    /// Buffer of right rows with same key (for many-to-many joins).
    right_buffer: Vec<Row>,
    /// Position in right buffer.
    buffer_position: usize,
    /// Whether current left row has been matched.
    matched_left: bool,
    /// Buffer of unmatched right rows for RIGHT/FULL outer joins.
    unmatched_right: Vec<Row>,
    /// Whether we're in the final phase of outputting unmatched right rows.
    outputting_unmatched: bool,
    /// Position in unmatched right rows.
    unmatched_position: usize,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl SortMergeJoinOp {
    /// Creates a new sort-merge join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        left_keys: Vec<LogicalExpr>,
        right_keys: Vec<LogicalExpr>,
        left: BoxedOperator,
        right: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(left.schema().merge(&right.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            left_keys,
            right_keys,
            filter: None,
            left,
            right,
            current_left: None,
            current_right: None,
            right_buffer: Vec::new(),
            buffer_position: 0,
            matched_left: false,
            unmatched_right: Vec::new(),
            outputting_unmatched: false,
            unmatched_position: 0,
            max_rows_in_memory: 0,
        }
    }

    /// Sets an additional filter condition.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Returns the join type.
    #[must_use]
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Compares keys from two rows.
    fn compare_keys(&self, left: &Row, right: &Row) -> OperatorResult<std::cmp::Ordering> {
        for (left_expr, right_expr) in self.left_keys.iter().zip(&self.right_keys) {
            let left_val = evaluate_expr(left_expr, left)?;
            let right_val = evaluate_expr(right_expr, right)?;

            let cmp = compare_values(&left_val, &right_val);
            if cmp != std::cmp::Ordering::Equal {
                return Ok(cmp);
            }
        }
        Ok(std::cmp::Ordering::Equal)
    }

    /// Checks if filter passes.
    fn filter_passes(&self, left: &Row, right: &Row) -> OperatorResult<bool> {
        match &self.filter {
            Some(f) => {
                let merged = left.merge(right);
                let result = evaluate_expr(f, &merged)?;
                Ok(matches!(result, Value::Bool(true)))
            }
            None => Ok(true),
        }
    }

    /// Creates a null row for the left side.
    fn left_null_row(&self) -> Row {
        Row::empty(self.left.schema())
    }

    /// Creates a null row for the right side.
    fn right_null_row(&self) -> Row {
        Row::empty(self.right.schema())
    }
}

impl Operator for SortMergeJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.left.open(ctx)?;
        self.right.open(ctx)?;
        self.current_left = self.left.next()?;
        self.current_right = self.right.next()?;
        self.right_buffer.clear();
        self.buffer_position = 0;
        self.matched_left = false;
        self.unmatched_right.clear();
        self.outputting_unmatched = false;
        self.unmatched_position = 0;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Phase: Output unmatched right rows for RIGHT/FULL outer joins
        if self.outputting_unmatched {
            if self.unmatched_position < self.unmatched_right.len() {
                let right_row = &self.unmatched_right[self.unmatched_position];
                self.unmatched_position += 1;
                self.base.inc_rows_produced();
                return Ok(Some(self.left_null_row().merge(right_row)));
            }
            self.base.set_finished();
            return Ok(None);
        }

        loop {
            // Phase 1: Return from buffer if available
            while self.buffer_position < self.right_buffer.len() {
                let left = match &self.current_left {
                    Some(l) => l.clone(),
                    None => break,
                };
                let right = self.right_buffer[self.buffer_position].clone();
                self.buffer_position += 1;

                if self.filter_passes(&left, &right)? {
                    self.matched_left = true;
                    self.base.inc_rows_produced();
                    return Ok(Some(left.merge(&right)));
                }
            }

            // Phase 2: Handle end of buffer - decide what to do with current left
            if !self.right_buffer.is_empty() && self.buffer_position >= self.right_buffer.len() {
                // Finished processing buffer for current left row
                // Check if we need to output unmatched left
                if !self.matched_left {
                    if let Some(left) = &self.current_left {
                        if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                            self.matched_left = true;
                            self.base.inc_rows_produced();
                            return Ok(Some(left.merge(&self.right_null_row())));
                        }
                    }
                }

                // Advance to next left row
                self.current_left = self.left.next()?;
                self.matched_left = false;
                self.buffer_position = 0;

                // Check if new left row has same key as buffer
                if let Some(left) = &self.current_left {
                    if let Some(first_right) = self.right_buffer.first() {
                        if self.compare_keys(left, first_right)? == std::cmp::Ordering::Equal {
                            // Same key - reprocess buffer
                            continue;
                        }
                    }
                }

                // Different key or no left - clear buffer and continue to main loop
                self.right_buffer.clear();
            }

            // Phase 3: Main merge loop - need both sides
            let (left, right) = match (&self.current_left, &self.current_right) {
                (Some(l), Some(r)) => (l, r),
                (None, Some(_)) => {
                    // Left exhausted, collect remaining right for RIGHT/FULL joins
                    if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                        if let Some(r) = self.current_right.take() {
                            self.unmatched_right.push(r);
                        }
                        while let Some(r) = self.right.next()? {
                            self.unmatched_right.push(r);
                        }
                        self.outputting_unmatched = true;
                        // Return first unmatched right row if any
                        if !self.unmatched_right.is_empty() {
                            let right_row = &self.unmatched_right[0];
                            self.unmatched_position = 1;
                            self.base.inc_rows_produced();
                            return Ok(Some(self.left_null_row().merge(right_row)));
                        }
                    }
                    self.base.set_finished();
                    return Ok(None);
                }
                (Some(left), None) => {
                    // Right exhausted, handle remaining left for LEFT/FULL joins
                    if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                        self.base.inc_rows_produced();
                        let result = left.merge(&self.right_null_row());
                        self.current_left = self.left.next()?;
                        return Ok(Some(result));
                    }
                    self.base.set_finished();
                    return Ok(None);
                }
                (None, None) => {
                    // Before finishing, check if we have unmatched rights to output
                    if matches!(self.join_type, JoinType::Right | JoinType::Full)
                        && !self.unmatched_right.is_empty()
                    {
                        self.outputting_unmatched = true;
                        self.unmatched_position = 0;
                        // Return first unmatched right row
                        let right_row = &self.unmatched_right[0];
                        self.unmatched_position = 1;
                        self.base.inc_rows_produced();
                        return Ok(Some(self.left_null_row().merge(right_row)));
                    }
                    self.base.set_finished();
                    return Ok(None);
                }
            };

            match self.compare_keys(left, right)? {
                std::cmp::Ordering::Less => {
                    // Left key < right key: advance left
                    // For LEFT/FULL joins, output unmatched left
                    if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                        self.base.inc_rows_produced();
                        let result = left.merge(&self.right_null_row());
                        self.current_left = self.left.next()?;
                        return Ok(Some(result));
                    }
                    self.current_left = self.left.next()?;
                }
                std::cmp::Ordering::Greater => {
                    // Left key > right key: advance right
                    // For RIGHT/FULL joins, track unmatched right
                    if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                        self.unmatched_right.push(right.clone());
                    }
                    self.current_right = self.right.next()?;
                }
                std::cmp::Ordering::Equal => {
                    // Keys match: buffer all right rows with same key
                    self.right_buffer.clear();
                    self.right_buffer.push(right.clone());

                    // Read more right rows with same key
                    loop {
                        self.current_right = self.right.next()?;
                        match &self.current_right {
                            Some(r) if self.compare_keys(left, r)? == std::cmp::Ordering::Equal => {
                                self.right_buffer.push(r.clone());

                                if self.max_rows_in_memory > 0
                                    && self.right_buffer.len() > self.max_rows_in_memory
                                {
                                    return Err(ParseError::QueryTooLarge {
                                        actual: self.right_buffer.len(),
                                        limit: self.max_rows_in_memory,
                                    });
                                }
                            }
                            _ => break,
                        }
                    }

                    self.buffer_position = 0;
                    self.matched_left = false;
                    // Loop will continue and process buffer in Phase 1
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.left.close()?;
        self.right.close()?;
        self.right_buffer.clear();
        self.unmatched_right.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "SortMergeJoin"
    }
}

/// Encodes a value into bytes for key comparison.
fn encode_key_value(value: &Value, buf: &mut Vec<u8>) {
    match value {
        Value::Null => buf.push(0),
        Value::Bool(b) => {
            buf.push(1);
            buf.push(u8::from(*b));
        }
        Value::Int(i) => {
            buf.push(2);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            buf.push(3);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::String(s) => {
            buf.push(4);
            buf.extend_from_slice(s.as_bytes());
            buf.push(0);
        }
        _ => buf.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_left() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Alice")],
                vec![Value::Int(2), Value::from("Bob")],
                vec![Value::Int(3), Value::from("Carol")],
            ],
        ))
    }

    fn make_right() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["user_id".to_string(), "order".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Order1")],
                vec![Value::Int(1), Value::from("Order2")],
                vec![Value::Int(2), Value::from("Order3")],
            ],
        ))
    }

    #[test]
    fn nested_loop_inner_join() {
        let condition = LogicalExpr::column("id").eq(LogicalExpr::column("user_id"));
        let mut op =
            NestedLoopJoinOp::new(JoinType::Inner, Some(condition), make_left(), make_right());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Alice has 2 orders, Bob has 1 order, Carol has 0
        assert_eq!(results.len(), 3);

        op.close().unwrap();
    }

    #[test]
    fn nested_loop_cross_join() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["a".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        ));
        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["b".to_string()],
            vec![vec![Value::Int(10)], vec![Value::Int(20)]],
        ));

        let mut op = NestedLoopJoinOp::new(JoinType::Cross, None, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while op.next().unwrap().is_some() {
            count += 1;
        }

        assert_eq!(count, 4); // 2 * 2
        op.close().unwrap();
    }

    #[test]
    fn hash_join_inner() {
        let build_keys = vec![LogicalExpr::column("id")];
        let probe_keys = vec![LogicalExpr::column("user_id")];

        let mut op = HashJoinOp::new(
            JoinType::Inner,
            build_keys,
            probe_keys,
            None,
            make_left(),
            make_right(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        assert_eq!(results.len(), 3);
        op.close().unwrap();
    }

    #[test]
    fn merge_join_inner() {
        // Sorted inputs
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));
        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(4)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = MergeJoinOp::new(JoinType::Inner, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Matches: 1, 2 (3 and 4 don't match)
        assert_eq!(results.len(), 2);
        op.close().unwrap();
    }

    // ========== IndexNestedLoopJoinOp Tests ==========

    #[test]
    fn index_nested_loop_join_inner() {
        // Outer: orders with user_id
        let outer: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["user_id".to_string(), "order".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Order1")],
                vec![Value::Int(1), Value::from("Order2")],
                vec![Value::Int(2), Value::from("Order3")],
                vec![Value::Int(99), Value::from("OrderUnmatched")],
            ],
        ));

        // Inner: users table (indexed by id)
        let inner: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Alice")],
                vec![Value::Int(2), Value::from("Bob")],
                vec![Value::Int(3), Value::from("Carol")],
            ],
        ));

        let outer_key = LogicalExpr::column("user_id");
        let inner_key = LogicalExpr::column("id");

        let mut op = IndexNestedLoopJoinOp::new(
            JoinType::Inner,
            outer_key,
            inner_key,
            "idx_users_id",
            outer,
            inner,
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // 3 matches: Order1->Alice, Order2->Alice, Order3->Bob
        // OrderUnmatched has no match
        assert_eq!(results.len(), 3);
        assert_eq!(op.index_name(), "idx_users_id");
        op.close().unwrap();
    }

    #[test]
    fn index_nested_loop_join_left_outer() {
        let outer: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["user_id".to_string(), "order".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Order1")],
                vec![Value::Int(99), Value::from("OrderUnmatched")],
            ],
        ));

        let inner: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string()],
            vec![vec![Value::Int(1), Value::from("Alice")]],
        ));

        let outer_key = LogicalExpr::column("user_id");
        let inner_key = LogicalExpr::column("id");

        let mut op = IndexNestedLoopJoinOp::new(
            JoinType::Left,
            outer_key,
            inner_key,
            "idx_users_id",
            outer,
            inner,
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // 2 results: Order1->Alice, OrderUnmatched->NULL
        assert_eq!(results.len(), 2);
        op.close().unwrap();
    }

    #[test]
    fn index_nested_loop_join_with_filter() {
        let outer: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["user_id".to_string(), "amount".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(100)],
                vec![Value::Int(1), Value::Int(50)],
                vec![Value::Int(2), Value::Int(200)],
            ],
        ));

        let inner: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "min_amount".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(75)],  // Alice requires >= 75
                vec![Value::Int(2), Value::Int(150)], // Bob requires >= 150
            ],
        ));

        let outer_key = LogicalExpr::column("user_id");
        let inner_key = LogicalExpr::column("id");
        let filter = LogicalExpr::column("amount").gt_eq(LogicalExpr::column("min_amount"));

        let mut op = IndexNestedLoopJoinOp::new(
            JoinType::Inner,
            outer_key,
            inner_key,
            "idx_users_id",
            outer,
            inner,
        )
        .with_filter(filter);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Only 2 matches: (100 >= 75), (200 >= 150)
        // (50 < 75) fails filter
        assert_eq!(results.len(), 2);
        op.close().unwrap();
    }

    // ========== SortMergeJoinOp Tests ==========

    #[test]
    fn sort_merge_join_inner() {
        // Sorted left input
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Alice")],
                vec![Value::Int(2), Value::from("Bob")],
                vec![Value::Int(3), Value::from("Carol")],
            ],
        ));

        // Sorted right input
        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["user_id".to_string(), "order".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Order1")],
                vec![Value::Int(1), Value::from("Order2")],
                vec![Value::Int(2), Value::from("Order3")],
            ],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("user_id")];

        let mut op = SortMergeJoinOp::new(JoinType::Inner, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Alice has 2 orders, Bob has 1 order, Carol has 0
        assert_eq!(results.len(), 3);
        assert_eq!(op.join_type(), JoinType::Inner);
        op.close().unwrap();
    }

    #[test]
    fn sort_merge_join_left_outer() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));

        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(3)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = SortMergeJoinOp::new(JoinType::Left, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // 3 results: 1->1, 2->NULL, 3->3
        assert_eq!(results.len(), 3);
        op.close().unwrap();
    }

    #[test]
    fn sort_merge_join_right_outer() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(3)]],
        ));

        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = SortMergeJoinOp::new(JoinType::Right, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // 3 results: 1->1, NULL->2, 3->3
        assert_eq!(results.len(), 3);
        op.close().unwrap();
    }

    #[test]
    fn sort_merge_join_full_outer() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(3)], vec![Value::Int(5)]],
        ));

        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(2)], vec![Value::Int(3)], vec![Value::Int(4)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = SortMergeJoinOp::new(JoinType::Full, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // 5 results: 1->NULL, NULL->2, 3->3, NULL->4, 5->NULL
        assert_eq!(results.len(), 5);
        op.close().unwrap();
    }

    #[test]
    fn sort_merge_join_many_to_many() {
        // Both sides have duplicates for key=1
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "l_val".to_string()],
            vec![
                vec![Value::Int(1), Value::from("A")],
                vec![Value::Int(1), Value::from("B")],
                vec![Value::Int(2), Value::from("C")],
            ],
        ));

        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "r_val".to_string()],
            vec![
                vec![Value::Int(1), Value::from("X")],
                vec![Value::Int(1), Value::from("Y")],
                vec![Value::Int(3), Value::from("Z")],
            ],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = SortMergeJoinOp::new(JoinType::Inner, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // 4 results: (A,X), (A,Y), (B,X), (B,Y)
        // 2 from left * 2 from right = 4 matches
        assert_eq!(results.len(), 4);
        op.close().unwrap();
    }

    #[test]
    fn sort_merge_join_with_filter() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "l_val".to_string()],
            vec![vec![Value::Int(1), Value::Int(10)], vec![Value::Int(2), Value::Int(20)]],
        ));

        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "r_val".to_string()],
            vec![vec![Value::Int(1), Value::Int(5)], vec![Value::Int(2), Value::Int(25)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];
        let filter = LogicalExpr::column("l_val").gt(LogicalExpr::column("r_val"));

        let mut op = SortMergeJoinOp::new(JoinType::Inner, left_keys, right_keys, left, right)
            .with_filter(filter);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Only 1 match: id=1 (10 > 5)
        // id=2 (20 < 25) fails filter
        assert_eq!(results.len(), 1);
        op.close().unwrap();
    }

    #[test]
    fn sort_merge_join_empty_inputs() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(vec!["id".to_string()], vec![]));

        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = SortMergeJoinOp::new(JoinType::Inner, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // No results from empty left
        assert_eq!(results.len(), 0);
        op.close().unwrap();
    }
}
